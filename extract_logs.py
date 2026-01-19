import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_recursive(root_dir, output_file="all_events_combined.csv"):
    all_data = []

    # Walk through the folder structure
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "events.out.tfevents" in file:
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    event_acc = EventAccumulator(file_path)
                    event_acc.Reload()
                    
                    for tag in event_acc.Tags()["scalars"]:
                        events = event_acc.Scalars(tag)
                        for e in events:
                            all_data.append({
                                "folder": root,
                                "metric": tag,
                                "value": e.value,
                                "step": e.step,
                                "wall_time": e.wall_time
                            })
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    # Convert list of dicts to DataFrame for better performance
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! Saved all data to {output_file}")
    else:
        print("No tfevents files found.")

# Run the extraction starting from your current directory
extract_recursive("/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/logs")