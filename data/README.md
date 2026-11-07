# Data

The dataset was created for this project by researching each smartphone individually; it is not a ready-made dataset.

## Sources

Specifications, benchmark scores, prices, and availability were collected from:

- [GSMArena](https://www.gsmarena.com)
- [91mobiles](https://www.91mobiles.com)
- official brand websites
- X (Twitter)
- YouTube videos covering the corresponding smartphones

The dataset is maintained over time as prices and availability change. Last updated: November 2026.

## Files

| File | Created by | Contents |
|---|---|---|
| `raw/Phones.csv` | manual research | The original dataset: 289 phones, 43 columns. Never modified by the notebooks. |
| `processed/phones_clean.csv` | notebook 01 | Cleaned data plus derived columns (camera-presence flags, launch year, phone age): 48 columns. |
| `processed/phones_with_value_scores.csv` | notebook 06 | Currently available phones with the expected price, value gap, and value score. |

The processed files are not stored in the repository; running the notebooks in order creates them.

## Overview

| Property | Value |
|---|---|
| Smartphones | 289 (155 currently available) |
| Brands | 15: Apple, Google, IQOO, Infinix, Lava, Motorola, Nothing, OnePlus, Oppo, POCO, Realme, Samsung, Tecno, Vivo, Xiaomi |
| Market | India; prices in Indian rupees (INR) |
| Launch dates | July 2022 to October 2026, mostly 2024–2026 |

## Column Definitions

### Identity and market

| Column | Description | Values / unit |
|---|---|---|
| `name` | Model name | text |
| `brand` | Manufacturer | 15 brands |
| `sale` | Sales channel or market tier | `Online`, `Offline`, `Both`, `Flagship` |
| `launch_date` | Launch date | `DD-MM-YYYY` |
| `price` | Price of the base variant (minimum RAM and storage) | INR |
| `availability` | Currently in stock in India | `1` = yes, `0` = no |
| `colors` | Number of colour options | count |

### Processor and memory

| Column | Description | Values / unit |
|---|---|---|
| `processor_type` | Chipset family | `Mediatek`, `Snapdragon`, `Exynos`, `Apple Bionic`, `Tensor`, `Unisoc` |
| `processor` | Chipset model | e.g. `Dimensity 7300`, `8 Elite`, `A18 Pro` |
| `fab` | Fabrication process | nm |
| `ram_type` | RAM technology | `LPDDR4X`, `LPDDR5`, `LPDDR5X` |
| `rom_type` | Storage technology | `UFS 2.2`, `UFS 3.1`, `UFS 4.0`, `UFS 4.1`, `NVMe` |

### Benchmarks

| Column | Description | Values / unit |
|---|---|---|
| `antutu` | AnTuTu benchmark score | points |
| `geekbench_single` | Geekbench single-core score | points |
| `geekbench_multiple` | Geekbench multi-core score | points |

### Display

| Column | Description | Values / unit |
|---|---|---|
| `display_type` | Panel type | `Amoled`, `IPS LCD` |
| `size` | Screen diagonal | inches |
| `curved_display` | Curved screen edges | `1` = yes, `0` = no |
| `refresh_rate` | Maximum refresh rate | Hz |
| `hbm` | Peak brightness (high-brightness mode) | nits |
| `ppi` | Pixel density | pixels per inch |
| `protection` | Screen protection glass | e.g. `Corning, GG 7i`, `Panda Glass` |

### Battery and charging

| Column | Description | Values / unit |
|---|---|---|
| `battery` | Battery capacity | mAh |
| `charging_speed` | Maximum wired charging power | W |
| `wireless_charging` | Wireless charging support | `1` = yes, `0` = no |

### Cameras

| Column | Description | Values / unit |
|---|---|---|
| `camera_number` | Total number of rear cameras | count |
| `useful_camera_number` | Rear cameras excluding low-resolution depth or macro lenses | count |
| `main` | Main camera resolution | MP |
| `camera_sensor_name` | Main camera sensor | e.g. `Sony IMX 882`, `Sony LYT 700` |
| `ultrawide` | Ultrawide camera resolution; empty if there is none | MP |
| `third` | Third rear camera type | `Telephoto - Nx` (N = optical zoom), `Macro`, `Useless` (low-resolution lens) |
| `third_mp` | Third camera resolution | MP |
| `fourth` | Fourth rear camera type | same format as `third` |
| `fourth_mp` | Fourth camera resolution | MP |
| `rear_video` | Best rear video mode as `resolution, frames per second` | e.g. `4, 60` = 4K at 60 fps; resolution `1` = 1080p, `2` = 2K, `0.44` = 480p |
| `front` | Front camera resolution | MP |
| `front_video` | Best front video mode | same format as `rear_video` |

### Build and audio

| Column | Description | Values / unit |
|---|---|---|
| `thickness` | Body thickness | mm |
| `weight` | Weight | g |
| `back_material` | Back panel material | `Plastic`, `Glass`, `Leather`, `Aluminium` |
| `frame_material` | Frame material | `Plastic`, `Aluminium`, `Titanium` |
| `ip_rating` | Dust and water resistance, as the two IP digits | e.g. `68` = IP68 |
| `speaker` | Speaker setup | `Mono`, `Stereo` |

## Missing Values

An empty cell means one of two things, which the analysis treats differently:

| Meaning | Columns | Missing |
|---|---|---|
| **The component does not exist** | `ultrawide` | 100 phones without an ultrawide camera |
| | `third`, `third_mp` | 93 / 94 phones without a third camera |
| | `fourth`, `fourth_mp` | 283 phones without a fourth camera |
| **The value is unknown** | `hbm` | 26 |
| | `protection` | 75 |
| | `camera_sensor_name` | 98 |

## Known Limitations

- **Base variants only.** Each phone is recorded with the price of its minimum RAM and storage variant.
- **Prices and availability are snapshots.** They reflect the time of the last update and change frequently.
- **Camera information is limited.** Sensor size, aperture, and image processing are not recorded, and the sensor name is unknown for about a third of the phones.
- **Not recorded:** NFC, USB port type, headphone jack, brand customer service, and software update policy.
- **Phones only.** Foldables and tablets are not included.
- **Indian market only.** The phone selection, prices, and availability reflect the Indian market.
