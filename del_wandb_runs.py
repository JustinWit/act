import wandb

# Replace with your API key or ensure you've logged in via wandb.login()
api = wandb.Api()

# Specify the project details
entity = "jwit3-georgia-institute-of-technology"  # Replace with your W&B username
project = "act-training"  # Replace with your W&B project name

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

# Iterate through the runs and delete those with "test" in their names
for run in runs:
    if "test" in run.name.lower():  # Check if "test" is in the run name
        print(f"Deleting run: {run.name} (ID: {run.id})")
        run.delete()
print("All test runs deleted.")
