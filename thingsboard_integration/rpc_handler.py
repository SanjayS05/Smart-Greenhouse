import requests
import time
import json
from threading import Thread

class ThingsBoardRPC:
    def __init__(self, tb_url, device_token):
        self.tb_url = tb_url
        self.device_token = device_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.actuator_states = {
            "fan": False,
            "light": False,
            "pump": False,
            "heater": False
        }
        self.running = False

    def process_command(self, command):
        """Handle incoming RPC commands"""
        method = command.get("method")
        params = command.get("params", {})
        
        print(f"Received RPC: {method} {params}")
        
        # Command router
        if method == "setFan":
            self.actuator_states["fan"] = params["on"]
            return {"success": True, "fan": self.actuator_states["fan"]}
            
        elif method == "setPump":
            self.actuator_states["pump"] = params["on"]
            return {"success": True, "pump": self.actuator_states["pump"]}
            
        elif method == "getStatus":
            return {"status": self.actuator_states}
            
        else:
            return {"error": "Unknown method"}

    def poll_rpc_commands(self):
        """Continuously check for new RPC commands"""
        while self.running:
            try:
                # Get pending RPCs
                url = f"{self.tb_url}/api/v1/{self.device_token}/rpc"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    for command in response.json():
                        result = self.process_command(command)
                        # Send response
                        requests.post(
                            f"{url}/{command['id']}",
                            headers=self.headers,
                            data=json.dumps(result)
                        )
                elif response.status_code != 404:
                    print(f"RPC poll error: {response.text}")
                    
            except Exception as e:
                print(f"RPC polling failed: {str(e)}")
                
            time.sleep(5)

    def start(self):
        """Start the RPC service"""
        self.running = True
        Thread(target=self.poll_rpc_commands, daemon=True).start()
        print("RPC handler started")

    def stop(self):
        """Stop the RPC service"""
        self.running = False
        print("RPC handler stopped")

# Example usage
if __name__ == "__main__":
    rpc = ThingsBoardRPC(
        tb_url="https://thingsboard.cloud",
        device_token="4ji4dhuuufdcabhwl6ww"
    )
    rpc.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rpc.stop()