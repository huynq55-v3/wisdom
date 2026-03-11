use rand::seq::{IteratorRandom, SliceRandom};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

fn main() {
    let base_dir = "./wisdom_models";

    // 1. Quét tìm tất cả các thư mục v0, v1, v2...
    let mut version_dirs: Vec<(usize, PathBuf)> = Vec::new();
    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('v') {
                        if let Ok(v) = name[1..].parse::<usize>() {
                            version_dirs.push((v, path));
                        }
                    }
                }
            }
        }
    }

    if version_dirs.is_empty() {
        println!("❌ Không tìm thấy thư mục data nào!");
        return;
    }

    // 🎯 THAY ĐỔI 1: Sắp xếp giảm dần (v lớn nhất đứng đầu)
    version_dirs.sort_by_key(|k| std::cmp::Reverse(k.0));

    // Rút thằng to nhất (version hiện tại) ra khỏi danh sách
    let (latest_v, latest_dir) = version_dirs.remove(0);

    println!("============================================================");
    println!("🛠️  CHUẨN BỊ DATASET CHO VERSION {}", latest_v + 1);
    println!("============================================================");
    println!("📦 Lấy toàn bộ data mới từ : v{}", latest_v);

    let mut latest_data = read_all_csv_in_dir(&latest_dir);
    println!("   ↳ Thu được: {} FENs", latest_data.len());

    // 🎯 THAY ĐỔI 2: Đặt giới hạn cho Replay Buffer (Cửa sổ trượt)
    let max_buffer_size = 1_000_000; // Giới hạn tối đa 1 triệu FEN cũ
    let mut buffer_data = Vec::new();

    // Quét từ version mới nhất lùi về v0
    for (v, dir) in &version_dirs {
        if buffer_data.len() >= max_buffer_size {
            break; // Đã no bụng, không ăn thêm data cũ nữa
        }

        println!("📚 Nạp Replay Buffer từ     : v{}", v);
        let mut data = read_all_csv_in_dir(dir);
        buffer_data.append(&mut data);
    }

    // 🎯 THAY ĐỔI 3: Chặt đứt phần đuôi nếu bị lố
    if buffer_data.len() > max_buffer_size {
        let excess = buffer_data.len() - max_buffer_size;
        println!(
            "✂️ Buffer vượt mức, đang xóa bỏ {} FENs cùi bắp nhất...",
            excess
        );
        buffer_data.truncate(max_buffer_size);
    }

    println!("   ↳ Tổng Replay Buffer hợp lệ: {} FENs", buffer_data.len());

    // 4. Bốc Random (Mix)
    let mut rng = rand::rng();
    let num_samples = 270_000; // Số FEN cũ muốn lấy để trộn

    let sampled_buffer = if buffer_data.len() > num_samples {
        println!("🎲 Đang bốc ngẫu nhiên {} FENs từ Buffer...", num_samples);
        buffer_data.into_iter().sample(&mut rng, num_samples)
    } else {
        println!("⚠️ Replay Buffer nhỏ hơn {}, lấy toàn bộ!", num_samples);
        buffer_data
    };

    // 5. Gộp và Xóc đều (Shuffle)
    let mut final_dataset = latest_data;
    final_dataset.extend(sampled_buffer);

    println!("🌪️ Đang xóc đều {} FENs tổng hợp...", final_dataset.len());
    final_dataset.shuffle(&mut rng);

    // 6. Ghi ra 1 file CSV duy nhất để up Kaggle
    let output_file = format!("{}/kaggle_train_v{}.csv", base_dir, latest_v + 1);
    let file = fs::File::create(&output_file).unwrap();
    let mut writer = BufWriter::new(file);

    for line in final_dataset {
        writeln!(writer, "{}", line).unwrap();
    }

    println!("\n✅ HOÀN TẤT! Đã xuất file: {}", output_file);
}

// Giữ nguyên hàm read_all_csv_in_dir như cũ...
fn read_all_csv_in_dir(dir: &PathBuf) -> Vec<String> {
    let mut lines = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("csv") {
                if let Ok(file) = fs::File::open(&path) {
                    let reader = BufReader::new(file);
                    for line in reader.lines().map_while(Result::ok) {
                        if !line.trim().is_empty() {
                            lines.push(line);
                        }
                    }
                }
            }
        }
    }
    lines
}
