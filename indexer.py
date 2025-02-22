import os
import yaml
import json

# Custom constructor to handle CloudFormation tags
def construct_undefined(self, node):
    if isinstance(node, yaml.ScalarNode):
        return node.value
    elif isinstance(node, yaml.SequenceNode):
        return [self.construct_object(child) for child in node.value]
    elif isinstance(node, yaml.MappingNode):
        return {self.construct_object(k): self.construct_object(v) for k, v in node.value}
    return node.value

yaml.SafeLoader.add_constructor(None, construct_undefined)

def build_index(root_dir="lambdas"):
    index = {"microservices": {}}
    
    for microservice in os.listdir(root_dir):
        microservice_path = os.path.join(root_dir, microservice)
        if not os.path.isdir(microservice_path):
            continue
            
        # Detect language
        languages = set()
        for root, _, files in os.walk(microservice_path):
            for file in files:
                if file.endswith(".js"):
                    languages.add("nodejs")
                elif file.endswith(".py"):
                    languages.add("python")
                elif file.endswith(".go"):
                    languages.add("golang")
        language = "nodejs" if "nodejs" in languages else (languages.pop() if languages else "unknown")
        
        # Parse serverless.yml with SafeLoader
        serverless_file = os.path.join(microservice_path, "serverless.yml")
        functions = []
        if os.path.exists(serverless_file):
            try:
                with open(serverless_file, "r") as f:
                    serverless_data = yaml.load(f, Loader=yaml.SafeLoader)
                if serverless_data and "functions" in serverless_data:
                    for func_name, func_data in serverless_data["functions"].items():
                        # Handle events safely
                        events = func_data.get("events", [])
                        trigger = "AppSync GraphQL" if not events else "unknown"
                        if isinstance(events, list) and events:
                            # If event is a dict (e.g., {http: {path: ...}}), extract key details
                            if isinstance(events[0], dict):
                                for event_type, details in events[0].items():
                                    if event_type == "http":
                                        trigger = f"http {details.get('method', 'unknown')} {details.get('path', 'unknown')}"
                                    else:
                                        trigger = str(event_type)
                                    break
                            else:
                                trigger = str(events[0])  # Fallback to string representation
                        functions.append({
                            "name": func_name,
                            "trigger": trigger,
                            "file": func_data.get("handler", "").split(".")[0] + ".js"  # Assuming Node.js
                        })
            except yaml.YAMLError as e:
                print(f"Warning: Could not parse {serverless_file} due to {e}. Skipping functions.")
        
        # Scan helpers
        helpers_dir = os.path.join(microservice_path, "helpers")
        helpers = []
        if os.path.exists(helpers_dir):
            for helper_file in os.listdir(helpers_dir):
                helpers.append({
                    "file": f"helpers/{helper_file}",
                    "purpose": f"Logic for {helper_file.split('.')[0]}"
                })
        
        # Scan controllers
        controllers_dir = os.path.join(microservice_path, "controllers")
        controllers = []
        if os.path.exists(controllers_dir):
            for controller_file in os.listdir(controllers_dir):
                controllers.append({
                    "file": f"controllers/{controller_file}",
                    "purpose": f"DB/API for {controller_file.split('.')[0]}"
                })
        
        index["microservices"][microservice] = {
            "language": language,
            "serverless_functions": functions,
            "helpers": helpers,
            "controllers": controllers
        }
    
    with open("index.json", "w") as f:
        json.dump(index, f, indent=2)
    print("Index built and saved to index.json")

if __name__ == "__main__":
    build_index()
