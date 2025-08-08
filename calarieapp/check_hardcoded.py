#!/usr/bin/env python3
"""
Script to check for hardcoded values in the codebase
"""

import os
import re
from typing import Dict, List, Tuple
from pathlib import Path

class HardcodedValueChecker:
    """Check for hardcoded values in Python files"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.hardcoded_values = {}
        
        # Patterns to look for
        self.patterns = {
            "numbers": {
                "calorie_targets": r"\b(2000|1000|5000)\b",
                "port_numbers": r"\b(8501|8080|3000)\b",
                "file_sizes": r"\b(10\s*\*\s*1024\s*\*\s*1024)\b",
                "timeouts": r"\b(3600|1800|300|60|30)\b",
                "retry_counts": r"\b(3|5|10)\b",
                "dimensions": r"\b(10|6|12|16|100)\b"
            },
            "strings": {
                "model_names": r"llama3-8b-8192|Salesforce/blip-image-captioning-base|yolov8n\.pt",
                "api_keys": r"GROQ_API_KEY",
                "urls": r"https?://[^\s\"']+",
                "file_paths": r"[a-zA-Z0-9_\-\./]+\.(pt|pkl|json|yaml|yml)",
                "dates": r"2025|2024",
                "names": r"Ujjwal Sinha",
                "github_links": r"github\.com/Ujjwal-sinha",
                "linkedin_links": r"linkedin\.com/in/sinhaujjwal01"
            },
            "colors": {
                "hex_colors": r"#[0-9A-Fa-f]{6}",
                "rgb_colors": r"rgb\([^)]+\)",
                "gradients": r"linear-gradient\([^)]+\)"
            }
        }
    
    def scan_file(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """Scan a single file for hardcoded values"""
        results = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for category, pattern_dict in self.patterns.items():
                    for pattern_name, pattern in pattern_dict.items():
                        matches = []
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line):
                                # Extract the matched value
                                match = re.search(pattern, line)
                                if match:
                                    matches.append((line_num, match.group()))
                        
                        if matches:
                            if category not in results:
                                results[category] = {}
                            results[category][pattern_name] = matches
                            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return results
    
    def scan_directory(self) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
        """Scan all Python files in the directory"""
        all_results = {}
        
        for file_path in self.root_dir.rglob("*.py"):
            if file_path.name != "check_hardcoded.py" and file_path.name != "config.py":
                print(f"Scanning: {file_path}")
                results = self.scan_file(file_path)
                if results:
                    all_results[str(file_path)] = results
        
        return all_results
    
    def generate_report(self, results: Dict[str, Dict[str, List[Tuple[int, str]]]]) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append("🔍 HARDCODED VALUES REPORT")
        report.append("=" * 50)
        report.append("")
        
        total_issues = 0
        
        for file_path, file_results in results.items():
            report.append(f"📁 File: {file_path}")
            report.append("-" * 30)
            
            file_issues = 0
            
            for category, patterns in file_results.items():
                for pattern_name, matches in patterns.items():
                    if matches:
                        report.append(f"  🔸 {category.upper()} - {pattern_name}:")
                        for line_num, value in matches:
                            report.append(f"    Line {line_num}: {value}")
                            file_issues += 1
                        report.append("")
            
            total_issues += file_issues
            report.append(f"  Total issues in file: {file_issues}")
            report.append("")
        
        report.append("=" * 50)
        report.append(f"📊 SUMMARY: {total_issues} hardcoded values found")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("1. Use config.py for all configuration values")
        report.append("2. Use environment variables for sensitive data")
        report.append("3. Use constants for magic numbers")
        report.append("4. Use configuration files for model paths")
        report.append("")
        
        return "\n".join(report)
    
    def check_config_usage(self) -> Dict[str, bool]:
        """Check if config.py is being used properly"""
        config_usage = {
            "app_config_imported": False,
            "model_config_imported": False,
            "nutrition_config_imported": False,
            "ui_config_imported": False
        }
        
        for file_path in self.root_dir.rglob("*.py"):
            if file_path.name != "config.py":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        if "from config import" in content or "import config" in content:
                            if "AppConfig" in content:
                                config_usage["app_config_imported"] = True
                            if "ModelConfig" in content:
                                config_usage["model_config_imported"] = True
                            if "NutritionConfig" in content:
                                config_usage["nutrition_config_imported"] = True
                            if "UIConfig" in content:
                                config_usage["ui_config_imported"] = True
                                
                except Exception as e:
                    print(f"Error checking config usage in {file_path}: {e}")
        
        return config_usage

def main():
    """Main function to run the hardcoded value check"""
    print("🔍 Starting hardcoded value check...")
    print("=" * 50)
    
    checker = HardcodedValueChecker()
    
    # Scan for hardcoded values
    print("📁 Scanning files for hardcoded values...")
    results = checker.scan_directory()
    
    # Generate report
    report = checker.generate_report(results)
    print(report)
    
    # Check config usage
    print("⚙️ Checking config.py usage...")
    config_usage = checker.check_config_usage()
    
    print("\n📋 CONFIG.PY USAGE STATUS:")
    for config_type, is_used in config_usage.items():
        status = "✅ Used" if is_used else "❌ Not Used"
        print(f"  {config_type}: {status}")
    
    # Save report to file
    with open("hardcoded_values_report.txt", "w") as f:
        f.write(report)
        f.write("\n\nCONFIG.PY USAGE STATUS:\n")
        for config_type, is_used in config_usage.items():
            status = "✅ Used" if is_used else "❌ Not Used"
            f.write(f"  {config_type}: {status}\n")
    
    print(f"\n📄 Report saved to: hardcoded_values_report.txt")
    
    # Summary
    total_files = len(results)
    total_issues = sum(
        sum(len(matches) for matches in file_results.values())
        for file_results in results.values()
    )
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Files scanned: {total_files}")
    print(f"  Hardcoded values found: {total_issues}")
    print(f"  Config.py usage: {sum(config_usage.values())}/{len(config_usage)}")
    
    if total_issues > 0:
        print(f"\n⚠️  ACTION REQUIRED: {total_issues} hardcoded values need to be moved to config.py")
    else:
        print(f"\n✅ EXCELLENT: No hardcoded values found!")

if __name__ == "__main__":
    main()
