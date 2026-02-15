import sys, json
data = json.load(sys.stdin)
gpus = [m for m in data if m.get('gpu_ram', 0) >= 40000]
gpus.sort(key=lambda x: x.get('dph_total', 999))
print("=== GPUs with 40GB+ VRAM ===")
for m in gpus:
    gpu = m.get('gpu_name', '?')
    n = m.get('num_gpus', 1)
    vram = m.get('gpu_ram', 0) / 1000
    total_vram = vram * n
    price = m.get('dph_total', 0)
    mid = m.get('id', '?')
    loc = m.get('geolocation', '?')
    cuda = m.get('cuda_max_good', '?')
    print(f"  {mid}  |  {n}x {gpu}  |  {total_vram:.0f}GB  |  ${price:.4f}/hr  |  CUDA {cuda}  |  {loc}")

# Also show 2x GPU setups
multi = [m for m in data if m.get('num_gpus', 1) >= 2 and m.get('gpu_ram', 0) >= 24000]
multi.sort(key=lambda x: x.get('dph_total', 999))
print("\n=== Multi-GPU (2+) with 24GB+ each ===")
for m in multi:
    gpu = m.get('gpu_name', '?')
    n = m.get('num_gpus', 1)
    vram = m.get('gpu_ram', 0) / 1000
    total_vram = vram * n
    price = m.get('dph_total', 0)
    mid = m.get('id', '?')
    loc = m.get('geolocation', '?')
    cuda = m.get('cuda_max_good', '?')
    print(f"  {mid}  |  {n}x {gpu}  |  {total_vram:.0f}GB total  |  ${price:.4f}/hr  |  CUDA {cuda}  |  {loc}")
