# 🐝 Swarm-OS: Decentralized Asymmetric Edge Inference
**Built for the AMD Slingshot Hackathon**

[![Demo Video](https://img.shields.io/badge/Watch-Live_Demo_Video-red?style=for-the-badge&logo=youtube)](#)<br>
[![PyPI](https://img.shields.io/badge/Install%20with-PIP-blue?style=for-the-badge&logo=pypi)](https://github.com/W-Samik/Swarm_OS)

**Swarm-OS** is a decentralized, peer-to-peer AI orchestration protocol. It allows massive Large Language Models (LLMs) to run across a fragmented network of consumer laptops over local Wi-Fi, completely bypassing the VRAM limits of monolithic GPUs. 

By implementing **Asymmetric Pipeline Parallelism**, **Distributed KV-Caching**, and **Explicit RoPE-Aligned Synchronization**, Swarm-OS fragments a neural network's layers, intelligently routes the tensors across an mDNS-discovered topology, and achieves near $O(1)$ inference speeds on standard edge hardware.

---

## 📦 Quick Install (Universal)

Swarm-OS is packaged as a globally executable Python library. You do not need to clone the repo or run scripts manually. 

**Run this command on both machines:**
```bash
pip install https://github.com/W-Samik/Swarm_OS/archive/main.zip
```
*(This method works on any Windows/Linux/Mac machine, even if Git is not installed).*

---

## 🎮 How to Run Swarm-OS

Once installed, the `swarm-os` command is available system-wide.

### 1. Start the Worker (Node B)
Run this on the first laptop. It will load the model, slice the neural network, and broadcast its availability via mDNS.
```bash
swarm-os --role B
```

### 2. Start the Master Orchestrator (Node A)
Run this on the second laptop. It will scan the LAN, lock onto the worker, and open the chat console.
```bash
swarm-os --role A
```

### 3. Advanced: Dynamic Model Switching
Swarm-OS supports multiple architectures. You can switch models using the `--model` flag:
```bash
# Use Qwen 1.5B (Smarter, AMD-Optimized)
swarm-os --role A --model qwen

# Use a custom HuggingFace ID
swarm-os --role A --model "microsoft/phi-2"
```
*(Supported presets: `tiny` (default), `qwen`, `solar`)*

---

## 🧠 The Deep Tech Architecture

Modern AI is bottlenecked by the "Memory Wall." To run a massive model locally, you traditionally need a $10,000+ GPU. Swarm-OS shatters this boundary by physically severing the Transformer block across a local area network (LAN).

1. **The mDNS Radar (`swarm_discovery.py`):** Nodes dynamically discover each other over local Wi-Fi using ZeroConf, negotiating a Reverse-IP Handshake. **No hardcoded IP addresses are required.**
2. **Asymmetric Sharding (`model_surgeon.py`):** The `nn.ModuleList` is dynamically sharded in RAM based on the detected hardware profile. 
   - **Node A (The Master):** Processes the prompt, computes Layers 0-10, and emits a microscopic 4KB Float16 Tensor.
   - **Node B (The Worker):** Receives the 4KB tensor over a zero-latency socket, computes Layers 11-21 + `lm_head`, and returns the predicted Token ID.
3. **Explicit RoPE Synchronization:** Physically severing a Transformer model across a network boundary normally causes Rotary Positional Embeddings (RoPE) to desynchronize, destroying the attention matrix and causing severe hallucinations. Swarm-OS explicitly synchronizes the sequence clock across the TCP pipe, guaranteeing **100% mathematical accuracy**.
4. **Zero-Latency Networking (`network_core.py`):** Utilizes `pyzmq` (ZeroMQ) with `TCP_NODELAY` to stream PyTorch tensors natively via `C_CONTIGUOUS` memory buffers, achieving sub-30ms round-trip latency.
---

## 📊 Live Telemetry & Visualization

Once connected, Swarm-OS visualizes the distributed compute in real-time.

### On Node A (The User Interface)
Node A explicitly announces its prefill phase and transmits the base tensor. It then streams the AI's response smoothly to the console using delta-printing.
```text
[Judge / User]: write a python script to print fibonacci

⚙️ [Node A Compute] Analyzing Prompt & Initializing KV-Cache...
🧠 [Node A Compute] Executing Layers 0-10 locally...
📤 [Node A Network] Sent 240.00KB Tensor to Node B over ZeroMQ.

💬 Swarm-AI: def fibonacci(n):
    a, b = 0, 1
    ...
[📉 Network Math: 142 x 4.09KB recursive tensors transmitted to Node B]
[⚡ Swarm Telemetry: 142 tokens generated in 19.20s | Speed: 7.39 Tokens/Sec]
```

### On Node B (The Hacker / Server Log)
Node B acts as the decentralized compute engine, rapidly processing the final 11 layers and returning the predicted tokens.
```text
[Node B] 🧹 Received RESET signal. Clearing KV-Cache & Penalties.
⚙️ [Node B] 📥 240.00KB | 🎯 Predicted: 'def'
⚙️ [Node B] 📥 4.09KB   | 🎯 Predicted: ' fib'
⚙️ [Node B] 📥 4.09KB   | 🎯 Predicted: 'onacci'
```
*(Notice how the KV-Cache drops the network payload from 240KB to just 4.09KB per token, achieving O(1) network efficiency!)*

---

## 🛠️ Troubleshooting: Connection Issues?

If Node A says `"Scanning local network..."` but never finds Node B:

**1. Allow Firewall Access:**
When you run the command for the first time, Windows/Mac will ask for network permission. You **MUST click "Allow"** (for Public/Private networks).

**2. The Ping Test:**
Try to ping Node B's IP from Node A. If it fails, open **Command Prompt as Administrator** on both machines and run:
```cmd
netsh advfirewall firewall add rule name="Swarm-OS Allow Ping" protocol=icmpv4:8,any dir=in action=allow
```

**3. The Hotspot Fix:**
If the venue Wi-Fi blocks peer-to-peer connections (AP Isolation), simply connect both laptops to a **Mobile Hotspot**. Swarm-OS uses local LAN traffic only and will **not** consume your cellular data.

---

## 📂 Repository Structure

| File | Purpose |
| :--- | :--- |
| `setup.py` | Configures the project as a globally installable `pip` package. |
| `swarm_os/cli.py` | The CLI entry point. Manages the Swarm Ring topology, memory wiping, and live terminal telemetry. |
| `swarm_os/network_core.py` | The low-latency `pyzmq` TCP transport layer. Handles atomic tensor serialization. |
| `swarm_os/model_surgeon.py` | The PyTorch brain. Handles dynamic `nn.ModuleList` slicing, explicit RoPE injection, and stochastic sampling. |
| `swarm_os/swarm_discovery.py`| The mDNS Radar. Broadcasts hardware profiles dynamically on the LAN. |

---

## 🔮 Future Roadmap: Enterprise Scalability

### 1. Zero-Waste Safetensor Streaming
Currently, both nodes cache the model file but execute only half. In V2, Swarm-OS will implement **Distributed Byte-Range Loading** via `safetensors`. Node A will HTTP-request *only* the bytes for Layers 0-10, and Node B will request Layers 11-21, allowing a 140GB model to run on laptops with small SSDs.

### 2. Scaling to *N*-Node Heterogeneous Pipelines
Swarm-OS natively supports scaling beyond a 2-node symmetric split. In V2, the mDNS discovery radar will automatically benchmark the hardware profile (CPU/GPU/NPU FLOPs) of $N$ discovered nodes. If the Master Orchestrator detects an **AMD Ryzen AI NPU** laptop alongside a weaker legacy laptop, it will dynamically shard the layers asymmetrically (e.g., 80% to AMD, 20% to legacy), creating a mathematically optimized Ring Topology.
