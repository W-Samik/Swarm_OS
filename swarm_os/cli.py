import numpy as np
import torch
import time
import sys
import json
import zmq
import argparse # NEW: For professional terminal commands

# NEW: Absolute imports based on the package name
from swarm_os.network_core import SwarmCommRouter
from swarm_os.model_surgeon import ShardedLlama
from swarm_os.swarm_discovery import SwarmDiscovery

def main():
    # Set up a professional command-line argument parser
    parser = argparse.ArgumentParser(description="Swarm-OS: Decentralized AI Inference")
    parser.add_argument(
        '--role', 
        choices=['A', 'B'], 
        required=True, 
        help="Specify 'A' for Master Node or 'B' for Worker Node."
    )
    args = parser.parse_args()
    role = args.role

    discovery = None

    try:
        if role == 'B':
            print("======================================")
            print("🟢 INITIALIZING SWARM WORKER (NODE B) 🟢")
            print("======================================")
            
            PORT_B = 8888
            discovery = SwarmDiscovery(node_id="Worker_B", port=PORT_B)
            discovery.broadcast_presence()
            
            net = SwarmCommRouter(node_id="B", listen_port=PORT_B)
            brain = ShardedLlama(role="B")
            
            print("\n[Node B] Broadcasting... Waiting for Master Node A to find me...")
            
            memory_cache_B = None
            seq_len_B = 0 # NEW: Track RoPE position
            
            while True:
                incoming_numpy = net.recv_tensor()
                
                # ... [Keep the handshake & connect-back logic the exact same] ...
                # --- AUTO-CONNECT BACK TO A ---
                if incoming_numpy.size == 5 and incoming_numpy[0] == -99.0:
                    ip_parts = [int(x) for x in incoming_numpy]
                    master_ip = f"{ip_parts[1]}.{ip_parts[2]}.{ip_parts[3]}.{ip_parts[4]}"
                    print(f"\n[Node B] Master Node A identified at {master_ip}. Connecting back...")
                    net.connect_to_next_node(master_ip, 7777)
                    net.send_tensor(np.array([-2.0], dtype=np.float16)) 
                    print("[Node B] Swarm Link Active. Waiting for AI Tensors...\n")
                    continue

                # --- MEMORY WIPE COMMAND ---
                if incoming_numpy.size == 1 and incoming_numpy[0] == -999.0:
                    memory_cache_B = None
                    seq_len_B = 0 # RESET THE CLOCK
                    net.send_tensor(np.array([-2.0], dtype=np.float16))
                    continue
                
                # --- AI INFERENCE (STATEFUL + ROPE SYNCED) ---
                if incoming_numpy.size > 10: 
                    device = brain.model.device
                    hidden_states_tensor = torch.from_numpy(incoming_numpy).to(dtype=torch.float16, device=device)
                    
                    # Process with memory AND RoPE position
                    token_id_numpy, memory_cache_B, seq_len_B = brain.process_node_B(
                        hidden_states_tensor, 
                        past_key_values=memory_cache_B,
                        current_seq_length=seq_len_B
                    )
                    net.send_tensor(token_id_numpy)

                    # --- 🌟 NEW: THE "HACKER" VISUALIZATION FOR NODE B 🌟 ---
                    # Decode the raw integer ID into an actual English word for the judges to see
                    predicted_word = brain.tokenizer.decode([int(token_id_numpy[0])])
                    
                    # Clean up formatting so newlines don't break our beautiful console logs
                    safe_word = predicted_word.replace('\n', '\\n').replace('\r', '')
                    payload_size = incoming_numpy.nbytes / 1024
                    
                    # Print the live server telemetry!
                    print(f"⚙️ [Node B Compute] 📥 Rcvd: {payload_size:.2f}KB Tensor | 🧠 Executed Layers 11-21 | 🎯 Predicted: '{safe_word}' | 📤 Sent to Node A")

        elif role == 'A':
            print("======================================")
            print("🔵 INITIALIZING SWARM MASTER (NODE A) 🔵")
            print("======================================")
            
            PORT_A = 7777
            net = SwarmCommRouter(node_id="A", listen_port=PORT_A)
            brain = ShardedLlama(role="A")
            
            print("\n[Node A] Activating mDNS Radar to find workers...")
            discovery = SwarmDiscovery(node_id="Master_A", port=PORT_A)
            found_nodes = discovery.search_for_nodes(timeout=5)
            
            if not found_nodes:
                print("\n❌ FATAL ERROR: No Swarm Workers found on the Wi-Fi.")
                sys.exit(1)
                
            target_ip = list(found_nodes.values())[0]['ip']
            target_port = list(found_nodes.values())[0]['port']
            
            print(f"\n[Node A] Successfully locked onto Worker at {target_ip}:{target_port}")
            net.connect_to_next_node(target_ip, target_port)
            
            print("\n[Node A] Sending Dynamic Handshake to Node B...")
            my_ip_parts = [float(x) for x in discovery.local_ip.split('.')]
            handshake_tensor = np.array([-99.0, my_ip_parts[0], my_ip_parts[1], my_ip_parts[2], my_ip_parts[3]], dtype=np.float16)
            
            handshake_successful = False
            while not handshake_successful:
                try:
                    header = {"dtype": str(handshake_tensor.dtype.name), "shape": list(handshake_tensor.shape)}
                    header_bytes = json.dumps(header).encode('utf-8')
                    net.sender.send_multipart([header_bytes, handshake_tensor.tobytes()], flags=zmq.NOBLOCK, copy=False)
                except zmq.Again: pass
                
                time.sleep(0.5)
                try:
                    reply = net.receiver.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                    reply_tensor = np.frombuffer(reply[1], dtype=np.float16)
                    if reply_tensor[0] == -2.0:
                        print("✅ Dynamic Swarm Link Fully Established!")
                        handshake_successful = True
                except zmq.Again: pass

            # --- THE MAGIC LLM LOOP (ACCURATE & STATELESS) ---
            print("\n======================================")
            print("       SWARM OS CONSOLE ACTIVE        ")
            print("  Type 'exit' to shut down the swarm. ")
            print("======================================")
            
            while True:
                user_input = input("\n[Judge / User]: ")
                if user_input.lower() == 'exit': break
                
                # Wipe Node B's memory
                net.send_tensor(np.array([-999.0], dtype=np.float16))
                ack = net.recv_tensor()
                while ack.size > 1 or ack[0] != -2.0: ack = net.recv_tensor()

                # 2. PREPARE THE PROMPT
                prompt = (
                    f"<|system|>\nYou are Swarm-OS, an intelligent decentralized AI assistant. "
                    f"Write clean, accurate, and concise answers.</s>\n"
                    f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
                )
                
                memory_cache_A = None 
                input_data = prompt 
                seq_len_A = 0 
                generated_token_ids =[]
                previously_printed_text = "" 
                
                # --- Telemetry Trackers ---
                tokens_generated = 0
                start_time = time.perf_counter()

                print(f"\n⚙️ [Node A Compute] Analyzing Prompt & Initializing KV-Cache...")
                print(f"🧠 [Node A Compute] Executing Layers 0-10 locally...")

                # --- Dynamic Ceiling (Max 1024 tokens) ---
                for i in range(1024):
                    intermediate_tensor, memory_cache_A, seq_len_A = brain.process_node_A(
                        prompt_or_token=input_data, 
                        past_key_values=memory_cache_A, 
                        current_seq_length=seq_len_A
                    )
                    
                    tensor_numpy = intermediate_tensor.cpu().numpy().astype(np.float16)
                    payload_size = tensor_numpy.nbytes / 1024
                    
                    # 🌟 THE CLARITY LOG: Print the massive first payload size! 🌟
                    if i == 0:
                        print(f"📤[Node A Network] Sent {payload_size:.2f}KB Tensor to Node B over ZeroMQ.\n")
                        print(f"💬 Swarm-AI: ", end="", flush=True)

                    net.send_tensor(tensor_numpy)
                    
                    returned_token_id_numpy = net.recv_tensor()
                    while returned_token_id_numpy.size > 1 or returned_token_id_numpy[0] < 0:
                        returned_token_id_numpy = net.recv_tensor()
                    
                    new_id = int(returned_token_id_numpy[0])
                    generated_token_ids.append(new_id)
                    tokens_generated += 1
                    
                    full_decoded_text = brain.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    new_text_to_print = full_decoded_text[len(previously_printed_text):]
                    print(new_text_to_print, end="", flush=True)
                    previously_printed_text = full_decoded_text
                    
                    # The AI naturally emits this token when its answer is complete
                    if new_id == brain.tokenizer.eos_token_id: 
                        break
                        
                    input_data = new_id 

                # --- Print Structural Telemetry ---
                end_time = time.perf_counter()
                generation_time = end_time - start_time
                tps = tokens_generated / generation_time
                
                print(f"\n\n[📉 Network Math: {tokens_generated} x 4.09KB recursive tensors transmitted to Node B]")
                print(f"[⚡ Swarm Telemetry: {tokens_generated} tokens generated in {generation_time:.2f}s | Speed: {tps:.2f} Tokens/Sec]")

            print("\n\n[Node A] Session Closed.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'net' in locals():
            net.close()
        if discovery is not None:
            discovery.cleanup()

# This allows you to still test it locally if needed
if __name__ == "__main__":
    main()