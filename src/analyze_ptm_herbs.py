"""
Analyze PTM Dataset - Traditional Chinese Medicine Herbs
Extract and analyze herb frequency from prescriptions.
"""
import re
from collections import Counter
from typing import List, Tuple


def extract_herbs_from_prescription(prescription_text: str) -> List[str]:
    """
    Extract herb names from prescription text.

    Format: herb_name (preparation_method) dosage
    Example: "穿山甲（汤浸透，取甲锉碎，同热灰铛内慢火炒令黄色）五钱"

    Args:
        prescription_text: Right side of prescription (herbs with dosages)

    Returns:
        List of herb names
    """
    herbs = []

    # Split by spaces (herbs are separated by spaces)
    parts = prescription_text.strip().split()

    for part in parts:
        if not part:
            continue

        # First, remove parentheses and their contents (preparation methods)
        herb_name = re.sub(r'[（(].*?[）)]', '', part)

        # Remove all dosage patterns at the end
        # Chinese numbers + units
        herb_name = re.sub(r'[一二三四五六七八九十百千半]+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)
        herb_name = re.sub(r'[一二三四五六七八九十百千半]+[钱两分厘克斤枚个粒片丸分两][一二三四五六七八九十百千半]+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)

        # Arabic numbers + units
        herb_name = re.sub(r'\d+[钱两分厘克斤枚个粒片丸分两]$', '', herb_name)

        # Remove "各" prefix/suffix
        herb_name = re.sub(r'^各', '', herb_name)
        herb_name = re.sub(r'各$', '', herb_name)
        herb_name = re.sub(r'各.*', '', herb_name)

        # Remove standalone numbers (Chinese and Arabic)
        if re.match(r'^[一二三四五六七八九十百千半\d]+$', herb_name):
            continue

        # Clean up
        herb_name = herb_name.strip()

        # Filter: must be at least 2 Chinese characters
        if herb_name and len(herb_name) >= 2 and re.search(r'[\u4e00-\u9fff]', herb_name):
            # Additional filter: remove if contains only numbers/units
            if not re.match(r'^[\d一二三四五六七八九十百千半钱两分厘克斤枚个粒片丸]+$', herb_name):
                herbs.append(herb_name)

    return herbs


def analyze_ptm_dataset(prescriptions_path: str = "data/PTM/data/prescriptions.txt"):
    """
    Analyze PTM dataset to extract herb frequency.

    Args:
        prescriptions_path: Path to prescriptions.txt
    """
    print("=" * 80)
    print("PTM Dataset Analysis - Traditional Chinese Medicine Herbs")
    print("=" * 80)

    herb_counter = Counter()
    total_prescriptions = 0
    total_herb_mentions = 0

    print(f"\nReading prescriptions from: {prescriptions_path}")

    with open(prescriptions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            # Split by tab: symptoms \t herbs
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            symptoms = parts[0]
            herbs_text = parts[1]

            # Extract herbs
            herbs = extract_herbs_from_prescription(herbs_text)

            # Update counters
            for herb in herbs:
                herb_counter[herb] += 1
                total_herb_mentions += 1

            total_prescriptions += 1

            # Progress indicator
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} prescriptions...")

    print(f"\n✅ Analysis Complete!")
    print(f"\nDataset Statistics:")
    print(f"  Total prescriptions: {total_prescriptions:,}")
    print(f"  Total herb mentions: {total_herb_mentions:,}")
    print(f"  Unique herbs: {len(herb_counter):,}")
    print(f"  Average herbs per prescription: {total_herb_mentions / total_prescriptions:.2f}")

    # Get most common herbs
    most_common = herb_counter.most_common(100)

    print("\n" + "=" * 80)
    print("Top 50 Most Common Herbs (从多到少)")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Herb Name':<20} {'Count':>10} {'Percentage':>10}")
    print("-" * 80)

    for rank, (herb, count) in enumerate(most_common[:50], 1):
        percentage = (count / total_herb_mentions) * 100
        print(f"{rank:<6} {herb:<20} {count:>10,} {percentage:>9.2f}%")

    # Show distribution analysis
    print("\n" + "=" * 80)
    print("Herb Frequency Distribution")
    print("=" * 80)

    # Count herbs by frequency ranges
    freq_ranges = {
        "10,000+": sum(1 for _, count in herb_counter.items() if count >= 10000),
        "5,000-9,999": sum(1 for _, count in herb_counter.items() if 5000 <= count < 10000),
        "1,000-4,999": sum(1 for _, count in herb_counter.items() if 1000 <= count < 5000),
        "500-999": sum(1 for _, count in herb_counter.items() if 500 <= count < 1000),
        "100-499": sum(1 for _, count in herb_counter.items() if 100 <= count < 500),
        "10-99": sum(1 for _, count in herb_counter.items() if 10 <= count < 100),
        "1-9": sum(1 for _, count in herb_counter.items() if 1 <= count < 10),
    }

    print(f"\n{'Frequency Range':<20} {'Number of Herbs':>20}")
    print("-" * 80)
    for range_name, herb_count in freq_ranges.items():
        print(f"{range_name:<20} {herb_count:>20,}")

    # Top 20 herbs account for what percentage?
    top_20_total = sum(count for _, count in most_common[:20])
    top_20_pct = (top_20_total / total_herb_mentions) * 100

    top_50_total = sum(count for _, count in most_common[:50])
    top_50_pct = (top_50_total / total_herb_mentions) * 100

    print("\n" + "=" * 80)
    print("Coverage Analysis")
    print("=" * 80)
    print(f"\nTop 20 herbs account for: {top_20_pct:.2f}% of all herb mentions")
    print(f"Top 50 herbs account for: {top_50_pct:.2f}% of all herb mentions")

    # Save results
    output_file = "data/PTM/herb_frequency_analysis.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PTM Dataset - Herb Frequency Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total prescriptions: {total_prescriptions:,}\n")
        f.write(f"Total herb mentions: {total_herb_mentions:,}\n")
        f.write(f"Unique herbs: {len(herb_counter):,}\n\n")
        f.write("Rank\tHerb Name\tCount\tPercentage\n")

        for rank, (herb, count) in enumerate(herb_counter.most_common(), 1):
            percentage = (count / total_herb_mentions) * 100
            f.write(f"{rank}\t{herb}\t{count}\t{percentage:.2f}%\n")

    print(f"\n✅ Full results saved to: {output_file}")

    return herb_counter, total_prescriptions, total_herb_mentions


if __name__ == "__main__":
    analyze_ptm_dataset()
