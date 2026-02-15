import sys, json
data = json.load(sys.stdin)

# Sort by total VRAM descending, then price ascending
results = []
for m in data:
    n = m.get('num_gpus', 1)
    vram_each = m.get('gpu_ram', 0) / 1000
    total_vram = vram_each * n
    price = m.get('dph_total', 0)
    gpu = m.get('gpu_name', '?')
    cuda = m.get('cuda_max_good', '?')
    loc = m.get('geolocation', '?')
    mid = m.get('id', '?')
    dlspeed = m.get('inet_down', 0)
    reliability = m.get('reliability', 0) * 100
    results.append((total_vram, price, mid, n, gpu, vram_each, cuda, loc, dlspeed, reliability))

# Show 48GB+ first (dream tier)
print("=" * 100)
print("  TIER 1: 48GB+ VRAM (can train Lux-B 512x512 comfortably)")
print("=" * 100)
tier1 = sorted([r for r in results if r[0] >= 48], key=lambda x: x[1])
if tier1:
    for tv, price, mid, n, gpu, ve, cuda, loc, dl, rel in tier1[:15]:
        print(f"  {mid:<12}  {n}x {gpu:<18}  {tv:.0f}GB total  ${price:.4f}/hr  CUDA {cuda}  DL:{dl:.0f}Mbps  R:{rel:.0f}%  {loc}")
else:
    print("  No 48GB+ GPUs currently available")

# Show 32-48GB (workable with optimizations)
print()
print("=" * 100)
print("  TIER 2: 32-47GB VRAM (needs gradient checkpointing)")
print("=" * 100)
tier2 = sorted([r for r in results if 32 <= r[0] < 48], key=lambda x: x[1])
for tv, price, mid, n, gpu, ve, cuda, loc, dl, rel in tier2[:15]:
    print(f"  {mid:<12}  {n}x {gpu:<18}  {tv:.0f}GB total  ${price:.4f}/hr  CUDA {cuda}  DL:{dl:.0f}Mbps  R:{rel:.0f}%  {loc}")

# Show A100 / H100 class (if any)
print()
print("=" * 100)
print("  TIER 3: Datacenter GPUs (A100, H100, A6000, etc.)")
print("=" * 100)
dc_names = ['A100', 'H100', 'H200', 'A6000', 'A100_PCIE', 'A100_SXM4', 'RTX_6000Ada', 'L40', 'L40S']
tier3 = sorted([r for r in results if any(d in r[4] for d in dc_names)], key=lambda x: x[1])
if tier3:
    for tv, price, mid, n, gpu, ve, cuda, loc, dl, rel in tier3[:15]:
        print(f"  {mid:<12}  {n}x {gpu:<18}  {tv:.0f}GB total  ${price:.4f}/hr  CUDA {cuda}  DL:{dl:.0f}Mbps  R:{rel:.0f}%  {loc}")
else:
    print("  No datacenter GPUs currently available")

print()
print(f"Total offers scanned: {len(data)}")
print(f"Offers with 24GB+ VRAM: {len([r for r in results if r[0] >= 24])}")
print(f"Offers with 48GB+ VRAM: {len([r for r in results if r[0] >= 48])}")
print(f"Offers with 80GB+ VRAM: {len([r for r in results if r[0] >= 80])}")
