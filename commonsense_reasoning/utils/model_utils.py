import transformers
import torch
import torch.nn as nn
from tqdm import tqdm


from utils.svdlora_trainer import SvdloraTrainer
from utils.sculpt_trainer import SculptTrainer
def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def get_trainer(trainer_params):
    peft_type = trainer_params['model'].peft_config.peft_type
    if "SVDLORA" in peft_type.upper():
        trainer = SvdloraTrainer(**trainer_params)
    elif "SCULPT" in peft_type.upper():
        trainer = SculptTrainer(**trainer_params)
    else:
        trainer = transformers.Trainer(**trainer_params)

    return trainer

def make_sequential_train(model, freeze_module: str = "lora_B", unfreeze_module: str = "lora_A"):
    for name, module in model.named_modules():
        if freeze_module in name:
            module.weight.requires_grad = False
        elif unfreeze_module in name:
            module.weight.requires_grad = True

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name='', target_modules=[]):
    if target_modules:
        for target in target_modules:
            if name.endswith(target):
                return {}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def calibrate_svd_matrices(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat


@torch.no_grad()
def calibrate_svd_matrices_low_resource(model_name, model, calib_loader, dev):
    # 低资源占用版本
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    # 使用Catcher模块捕获输入输出
    # 不使用调用embedding层来生成第一层的输入，是为了适配不同的模型结构，有些模型在第一层之前
    # 存在不同的操作，使用Catcher可以通配不同结构的模型
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            # 保持原始 attention_mask 形状
                mask = kwargs['attention_mask'].cpu()
            else:
                # 生成默认的 attention_mask，形状为 (1, 1, seq_len, seq_len)
                batch_size, seq_len = inp.shape[0], inp.shape[1]
                mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=inp.dtype)

            if cache['attention_mask'] is None:
                cache['attention_mask'] = mask
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], mask), dim=0)

            if "opt" not in model_name and 'position_ids' in kwargs:
                if cache['position_ids'] is None:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
                else:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)

            raise ValueError
    # layers实际为model的所有层的引用，对layers中的层进行修改，就是对model中的层进行修改
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    # 捕获结束之后删除Catcher，重新移动到cpu上
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    # outs存储的是transformer的输出，由于add&norm的存在，transformer的输入输出维度相同，因此用zero_like来初始化outs
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}  # 每一层中所有线性层的从输入的协方差矩阵中得到的缩放矩阵
    for i in tqdm(range(len(layers))):
        layer_profile = {}  # layer_profile记录第i层中所有线性层的从输入的协方差矩阵中得到的缩放矩阵
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        # 初始化scaling_diag_matrix属性，用于存储协方差矩阵
        # 为当前层中的每一个线性层注册hook，存储到handles中，便于后续移除hook
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        # 遍历数据集，对每个数据进行前向传播
        # 在这个过程中，利用hook函数得到协方差矩阵
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        # 移除hook，复原linear层
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat

@torch.no_grad()
def whitening(model_name, model, profiling_mat, lora_r, target_modules, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                # 求逆矩阵
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            truc_s = S[:lora_r]
            truc_u = U[:, :lora_r]
            truc_v = torch.matmul(VT[:lora_r, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            for module_name in target_modules:
                if module_name in name:
                    subset[name].weight.lora_A.weight.data.copy_(svd_v.detach())
                    subset[name].weight.lora_B.weight.data.copy_(svd_u.detach())
                    weight_low = transpose(subset[name].weight.lora_B.weight @ subset[name].weight.lora_A.weight, subset[name].fan_in_fan_out) * subset[name].scaling

            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()
