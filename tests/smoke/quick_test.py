print("Hello from Python!")
import os
print("Current directory:", os.getcwd())
print("Dataset exists:", os.path.exists("dataset"))
print("Dataset periorbital exists:", os.path.exists("dataset/dataset_periorbital"))
if os.path.exists("dataset/dataset_periorbital"):
    ethnicities = os.listdir("dataset/dataset_periorbital")
    print("Ethnicities found:", ethnicities)
print("Test completed successfully!")

