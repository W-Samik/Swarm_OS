# 🐝 Swarm-OS: Decentralized Asymmetric Edge Inference
**Built for the AMD Slingshot Hackathon**

[![Demo Video](https://img.shields.io/badge/Watch-Live_Demo_Video-red?style=for-the-badge&logo=youtube)](#) *(<-- REPLACE THIS # WITH YOUR UNLISTED YOUTUBE LINK)*
[![GitHub Package](https://img.shields.io/badge/Install%20with-PIP-blue?style=for-the-badge&logo=pypi)](https://github.com/W-Samik/Swarm_OS)

**Swarm-OS** is a decentralized, peer-to-peer AI orchestration protocol. It allows massive Large Language Models (LLMs) to run across a fragmented network of consumer laptops over local Wi-Fi, completely bypassing the VRAM limits of monolithic GPUs. 

By implementing **Asymmetric Pipeline Parallelism**, **Distributed KV-Caching**, and **Explicit RoPE-Aligned Synchronization**, Swarm-OS fragments a neural network's layers, intelligently routes the tensors across an mDNS-discovered topology, and achieves near $O(1)$ inference speeds on standard edge hardware.

---

## 📦 Quick Install (Universal)

Swarm-OS is packaged as a globally executable Python library. You do not need to clone the repo manually. 

**Run this command on both machines:**
```bash
pip install https://github.com/W-Samik/Swarm_OS/archive/main.zip
```
*(This works on Windows, Linux, and Mac, requiring only a Python environment).*

---

## 🎮 How to Run Swarm-OS (CLI Mode)

Once installed, the `swarm-os` command is available system-wide.

### 1. Start the Worker (Node B)
Run this on the first machine. It loads the model, slices the network, and broadcasts its presence via mDNS.
```bash
swarm-os --role B
```

### 2. Start the Master Orchestrator (Node A)
Run this on the second machine. It will automatically scan the LAN, lock onto the worker, and open the chat console.
```bash
swarm-os --role A
```

---

## 🛠️ Python SDK Integration (Library Usage)

Swarm-OS is a full-fledged SDK. Developers can `import swarm_os` to build custom decentralized AI applications. Below is a sample script showing how to use the Master Node as a programmable library:

```python
import swarm_os
import time

# Initialize the Master Orchestrator
# Make sure to run (swarm-os --role B) on the worker node device before executing the code
bot = swarm_os.SwarmMaster(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

try:
    # Scan the local network and connect to available Swarm Workers
    bot.connect()
    
    questions = ["What is 2+2?", "Who is the CEO of AMD?"]
    
    for q in questions:
        print(f"\n[User]: {q}")
        print("[Swarm-AI]: ", end="", flush=True)
        
        start_time = time.perf_counter()
        
        # Use stream=True to generate text word-by-word
        for chunk in bot.generate(q, stream=True):
            print(chunk, end="", flush=True)
        
        # Access live telemetry from the bot instance
        end_time = time.perf_counter()
        gen_time = end_time - start_time
        tps = bot.total_tokens / gen_time
        
        print(f"\n\n[📉 Network Math: {bot.total_tokens} x 4.09KB recursive tensors transmitted]")
        print(f"[⚡ Swarm Telemetry: {bot.total_tokens} Tokens | Speed: {tps:.2f} Tokens/Sec]")
        print("-" * 30)
        
finally:
    bot.close()
```

---

## 🧠 The Deep Tech Architecture

Modern AI is bottlenecked by the "Memory Wall." Swarm-OS shatters this boundary by physically severing the Transformer block across a local area network (LAN).

1. **The mDNS Radar (`swarm_discovery.py`):** Nodes dynamically discover each other over local Wi-Fi using ZeroConf, negotiating a Reverse-IP Handshake. **No hardcoded IP setup is required.**
2. **Asymmetric Sharding (`model_surgeon.py`):** The `nn.ModuleList` is dynamically sharded in RAM. 
   - **Node A (The Master):** Computes Layers 0-10, emitting a microscopic 4KB Float16 Tensor representing the mathematical hidden states.
   - **Node B (The Worker):** Receives the tensor, computes Layers 11-21 + `lm_head`, and returns the predicted Token ID.
3. **Explicit RoPE Synchronization:** Severing a model normally causes Rotary Positional Embeddings (RoPE) to desynchronize, destroying the attention matrix. Swarm-OS explicitly synchronizes the sequence clock across the TCP pipe, guaranteeing **100% mathematical accuracy**.
4. **Zero-Latency Networking (`network_core.py`):** Utilizes `pyzmq` with `TCP_NODELAY` to stream tensors natively via memory buffers, achieving sub-30ms round-trip latency.

---

## 💻 AMD Hardware Integration Strategy

Swarm-OS is architecturally designed for the **AMD Heterogeneous Compute Ecosystem**. 

In a production deployment, the Swarm Orchestrator tasks (mDNS networking, ZeroMQ routing, and KV-Cache management) run natively at ultra-low wattage on the **AMD Ryzen AI NPU**. Simultaneously, the heavy `Float16` transformer layers are routed to idle **AMD Radeon GPUs** distributed across the local subnet. We turn a room full of thin-and-light laptops into a decentralized supercomputer.

---

## 📊 Live Telemetry & Visualization

Swarm-OS visualizes the distributed compute in real-time within the terminal.

### On Node A (Master)
Node A announces the prefill phase, local layer execution, and the final telemetry summary.
```text
⚙️ [Node A Compute] Analyzing Prompt & Initializing KV-Cache...
🧠 [Node A Compute] Executing Layers 0-10 locally...
📤 [Node A Network] Sent 240.00KB Tensor to Node B over ZeroMQ.

💬 Swarm-AI: The Allies won...
[📉 Network Math: 82 x 4.09KB recursive tensors transmitted to Node B]
[⚡ Swarm Telemetry: 82 tokens generated in 13.48s | Speed: 6.08 Tokens/Sec]
```

### On Node B (Worker)
Node B shows the hacker-style server log, proving it is receiving decentralized math thoughts.
```text
⚙️ [Node B Compute] 📥 Rcvd: 4.09KB | 🧠 Executed Layers 11-21 | 🎯 Predicted: ' Allied'
⚙️ [Node B Compute] 📥 Rcvd: 4.09KB | 🧠 Executed Layers 11-21 | 🎯 Predicted: ' Powers'
```

---

## 🛠️ Troubleshooting: Connection Issues?

**1. Allow Firewall Access:**
Windows/Mac Firewall will ask for network permission. You **MUST click "Allow"** for both Public and Private networks.

**2. The Ping Test:**
If nodes don't find each other, open **Command Prompt as Administrator** and run this to allow ICMP traffic:
```cmd
netsh advfirewall firewall add rule name="Swarm-OS Allow Ping" protocol=icmpv4:8,any dir=in action=allow
```

**3. The Hotspot Fix:**
If venue Wi-Fi blocks P2P traffic (AP Isolation), connect both laptops to a **Mobile Hotspot**. Swarm-OS uses local LAN traffic only and will **not** consume cellular data.

---

## 📂 Repository Structure

| File | Purpose |
| :--- | :--- |
| `setup.py` | Configures the project as a globally installable `pip` library. |
| `swarm_os/cli.py` | The Terminal entry point with telemetry and model flags. |
| `swarm_os/engine.py` | The Core SDK classes (`SwarmMaster`, `SwarmWorker`). |
| `swarm_os/network_core.py`| The low-latency TCP transport layer for tensors. |
| `swarm_os/model_surgeon.py`| Handles dynamic layer slicing and RoPE synchronization. |
| `swarm_os/swarm_discovery.py`| The mDNS Radar for auto-discovery. |

---

## 🔮 Future Roadmap: Enterprise & Research

### V2.0: Scaling & Efficiency
- **Feature 1: Zero-Waste Safetensor Streaming:** Currently, nodes download the full model. In V2, we will use **HTTP Range Requests** to download *only* the specific byte-ranges from Hugging Face representing a node's assigned layers. This allows a 140GB model to run on a laptop with a 128GB SSD by distributing the storage footprint.
- **Feature 2: N-Node Ring Topology:** Transitioning from 2-node "Ping-Pong" to an $N$-node "Ring" ($A \rightarrow B \rightarrow C \rightarrow A$). This enables nearly infinite scaling of model size by pooling hardware.

### V3.0: Research-Level Innovations
- **Feature 3: Speculative Decoding over LAN:** We will load a tiny "Draft Model" (50M params) on Node A. Node A guesses 5 words locally and sends a "Guess Bundle" to Node B. Node B verifies all 5 words in a single parallel pass, effectively hiding network latency.
- **Feature 4: Dynamic Load Balancing:** Real-time profiling of thermal throttling. If Node B slows down, the Orchestrator dynamically shifts layers back to Node A to maintain maximum throughput.
- **Feature 5: Self-Healing Agentic Redistribution:**
  - **Fault Tolerance:** An RL-based Orchestrator will detect if a node disconnects and instantly redistribute the workload across surviving nodes flawlessly.
  - **Dynamic Joining:** If a new node joins mid-sentence, the system re-profiles the network and redistributes layers to the new device to increase speed without interrupting the user.
