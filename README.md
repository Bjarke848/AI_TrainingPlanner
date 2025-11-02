# 🤖 AI Training Planner for Garmin Data

> Turn your Garmin training data into AI-powered weekly training plans.

---

## 🚀 Setup (Once)

1. 📥 Export data from [Garmin Connect](https://connect.garmin.com) → Activities → CSV
2. 💾 Save as `Activities.csv` in this folder
3. 🐍 Install: `pip install pandas numpy pyperclip`
4. ⚙️ Run: `python garmin_data_processor.py setup`
5. 🎯 Enter your training goals (saved to `my_training_goals.txt`)

## 📅 Weekly Workflow

**Quick Method (Recommended):**
```bash
python garmin_data_processor.py feedback
# Answer quick questions → Prompt auto-copied to clipboard → Paste in AI!
```

**Traditional Method:**
1. 📥 Export fresh `Activities.csv` from Garmin
2. 📝 (Optional) Edit `body_feedback.txt` or run `feedback` command
3. ▶️ Run: `python garmin_data_processor.py weekly`
4. ✅ **Prompt auto-copied to clipboard!**
5. 🤖 Paste (Ctrl+V) into ChatGPT/Claude/Copilot
6. 📊 Get your personalized weekly plan!

## ✨ Smart Features

- 📋 **Auto-copy to clipboard** - No manual copying needed!
- 🏁 **Race proximity alerts** - Color-coded warnings (🔴 1 week, 🟠 2 weeks, 🟢 5-8 weeks)
- ⚠️ **Training load warnings** - Auto-detect injury risk from A/C ratio
- 💬 **Interactive feedback** - Quick CLI prompts instead of editing files

## 📁 Files

| File | Purpose | Update? |
|------|---------|---------|
| `Activities.csv` | 📊 Garmin export | ✅ Weekly |
| `my_training_goals.txt` | 🎯 Goals & constraints | 📝 As needed |
| `body_feedback.txt` | 💪 Weekly check-in | 🔄 Optional |
| `weekly_plan_YYYYMMDD.txt` | 📄 AI prompts | 🤖 Auto-saved |

## 🔍 What's Analyzed

- 📈 4-week training history & patterns
- ⚡ 7-day & 28-day training load (acute/chronic ratio with auto-warnings)
- 🏃‍♂️ Recent ultra runs or high-intensity sessions
- 🏁 Race calendar with proximity alerts (taper/peak/build phases)
- 💭 Current body feedback (fatigue, soreness)

## 💡 Pro Tips

- ✅ Export **all** activities each week (not just new ones)
- 🎯 Use `feedback` command for fastest updates
- ⚠️ Watch for 🔴 warnings - they indicate injury risk
- 🏁 Race alerts help AI taper you properly
- 🔄 Update goals when races or priorities change

---

**Happy Training!** 🏃‍♂️ 🚴‍♂️ 🏊‍♂️ 💪
