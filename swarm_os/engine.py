import torch
import numpy as np
import time
import zmq
from .network_core import SwarmCommRouter
from .model_surgeon import ShardedLlama
from .swarm_discovery import SwarmDiscovery

class SwarmWorker:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=8888):
        print(f"🟢 Initializing Swarm Worker (Port {port})...")
        self.brain = ShardedLlama(model_id=model_id, role="B")
        self.net = SwarmCommRouter(node_id="B", listen_port=port)
        self.discovery = SwarmDiscovery(node_id="Worker_B", port=port)
        self.memory_cache = None
        self.seq_len = 0

    def start(self):
        """Starts the infinite blocking listener loop."""
        self.discovery.broadcast_presence()
        print("[Node B] Broadcasting... Waiting for Master...")
        
        try:
            while True:
                incoming_numpy = self.net.recv_tensor()
                
                # Handshake / Reset Logic
                if incoming_numpy.size == 1 and incoming_numpy[0] == -999.0:
                    self.memory_cache = None
                    self.seq_len = 0
                    self.net.send_tensor(np.array([-2.0], dtype=np.float16))
                    continue
                
                # Connection Logic
                if incoming_numpy.size == 5 and incoming_numpy[0] == -99.0:
                    ip_parts = [int(x) for x in incoming_numpy]
                    master_ip = f"{ip_parts[1]}.{ip_parts[2]}.{ip_parts[3]}.{ip_parts[4]}"
                    print(f"\n[Node B] Connecting back to Master at {master_ip}...")
                    self.net.connect_to_next_node(master_ip, 7777)
                    self.net.send_tensor(np.array([-2.0], dtype=np.float16))
                    continue

                # Inference Logic
                if incoming_numpy.size > 10:
                    device = self.brain.model.device
                    hidden_states = torch.from_numpy(incoming_numpy).to(dtype=torch.float16, device=device)
                    
                    token_id, self.memory_cache, self.seq_len = self.brain.process_node_B(
                        hidden_states, 
                        past_key_values=self.memory_cache,
                        current_seq_length=self.seq_len
                    )
                    self.net.send_tensor(token_id)
                    
                    # Optional: Print for debugging
                    # word = self.brain.tokenizer.decode([int(token_id[0])])
                    # print(f"⚙️ Predicted: '{word.replace(chr(10), '')}'")

        except KeyboardInterrupt:
            print("\n[Node B] Stopping...")
        finally:
            self.net.close()
            self.discovery.cleanup()

class SwarmMaster:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", port=7777):
        print(f"🔵 Initializing Swarm Master (Port {port})...")
        self.brain = ShardedLlama(model_id=model_id, role="A")
        self.net = SwarmCommRouter(node_id="A", listen_port=port)
        self.discovery = SwarmDiscovery(node_id="Master_A", port=port)
        self.connected = False

    def connect(self, timeout=10):
        """Scans for a worker and establishes connection."""
        print("[Node A] Scanning for workers...")
        found_nodes = self.discovery.search_for_nodes(timeout=timeout)
        if not found_nodes:
            raise TimeoutError("No Swarm Workers found.")
        
        target_ip = list(found_nodes.values())[0]['ip']
        target_port = list(found_nodes.values())[0]['port']
        print(f"[Node A] Found worker at {target_ip}:{target_port}")
        
        self.net.connect_to_next_node(target_ip, target_port)
        
        my_ip = [float(x) for x in self.discovery.local_ip.split('.')]
        handshake = np.array([-99.0, my_ip[0], my_ip[1], my_ip[2], my_ip[3]], dtype=np.float16)
        
        print("[Node A] Sending Handshake...")
        self.net.send_tensor(handshake)
        self.connected = True
        print("✅ Connected!")

    def _internal_generator(self, user_prompt, max_tokens):
        """Hidden generator that does the actual math."""
        # 1. Send Reset Signal
        self.net.send_tensor(np.array([-999.0], dtype=np.float16))
        self.net.recv_tensor() # Wait for ACK

        # 2. Prepare Prompt
        formatted_prompt = (
            f"<|system|>\nYou are Swarm-OS. Answer accurately.</s>\n"
            f"<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        )
        
        memory_cache = None
        input_data = formatted_prompt
        seq_len = 0
        generated_ids = []
        prev_text = ""

        for _ in range(max_tokens):
            intermediate, memory_cache, seq_len = self.brain.process_node_A(
                prompt_or_token=input_data, 
                past_key_values=memory_cache, 
                current_seq_length=seq_len
            )
            
            self.net.send_tensor(intermediate.cpu().numpy().astype(np.float16))
            
            token_numpy = self.net.recv_tensor()
            while token_numpy.size > 1 or token_numpy[0] < 0:
                token_numpy = self.net.recv_tensor()
            
            new_id = int(token_numpy[0])
            generated_ids.append(new_id)
            
            full_decoded = self.brain.tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = full_decoded[len(prev_text):]
            prev_text = full_decoded
            
            yield new_text
            
            if new_id == self.brain.tokenizer.eos_token_id:
                break
            input_data = new_id

    def generate(self, user_prompt, max_tokens=300, stream=True):
        """
        Public API. 
        If stream=True, returns a generator.
        If stream=False, returns a fully concatenated string.
        """
        if not self.connected:
            raise ConnectionError("Not connected to Swarm. Call connect() first.")

        gen = self._internal_generator(user_prompt, max_tokens)
        
        if stream:
            return gen
        else:
            # Exhaust the generator to build the full string
            return "".join(list(gen))

    def close(self):
        self.net.close()
        self.discovery.cleanup()