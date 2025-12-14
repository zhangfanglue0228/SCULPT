import os
import argparse

def main():

    args = parse_args()

    result_file_path = os.path.join(args.lora_weights, "ALL_results.txt")
    
    if not os.path.exists(result_file_path):
        raise FileNotFoundError(f"Result file not found: {result_file_path}")
    
    accuracies = []
    with open(result_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() and not line.startswith("Average Accuracy:"):
                parts = line.strip().split(":")
                if len(parts) == 2:
                    try:
                        accuracy = float(parts[1].strip())
                        accuracies.append(accuracy)
                    except ValueError:
                        print(f"Warning: Could not parse accuracy value from line: {line}")
    
    if accuracies:
        average_accuracy = sum(accuracies) / len(accuracies)
        
        with open(result_file_path, "a") as f:
            f.write(f"\nAverage Accuracy: {average_accuracy}\n")
        
        print(f"Average accuracy calculated: {average_accuracy}")
    else:
        print("No valid accuracy values found in the file.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_weights', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()