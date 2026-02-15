import sys, json
data = json.load(sys.stdin)
gpus = [m for m in data if m.get('gpu_ram', 0) >= 24000]
gpus.sort(key=lambda x: x.get('dph_total', 999))
print("Available GPUs with 24GB+ VRAM (sorted by price):\n")
print(f"{'ID':<12} {'Config':<22} {'Total VRAM':<12} {'Price':<10} {'Location'}")
print("-" * 85)
for m in gpus[:25]:
    gpu = m.get('gpu_name', '?')
    n = m.get('num_gpus', 1)
    vram = m.get('gpu_ram', 0) / 1000
    total_vram = vram * n
    price = m.get('dph_total', 0)
    mid = m.get('id', '?')
    loc = m.get('geolocation', '?')
    config = f"{n}x {gpu}"
    print(f"{mid:<12} {config:<22} {total_vram:.0f} GB       ${price:.4f}/hr  {loc}")
