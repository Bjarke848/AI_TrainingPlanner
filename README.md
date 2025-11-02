# 🤖 AI Training Planner for Garmin Data

> Turn your Garmin training data into AI-powered weekly training plans.

---

## 🚀 Setup (Once)

1. 📥 Export data from [Garmin Connect](https://connect.garmin.com) → Activities → CSV
2. 💾 Save as `Activities.csv` in this folder
3. 🐍 Install: `pip install pandas numpy`
4. ⚙️ Run: `python garmin_data_processor.py setup`
5. 🎯 Enter your training goals (saved to `my_training_goals.txt`)

## 📅 Weekly Workflow
1. 📥 Export fresh `Activities.csv` from Garmin
2. 📝 (Optional) Update `body_feedback.txt` with current fatigue/soreness
3. ▶️ Run: `python garmin_data_processor.py weekly`
4. 📋 Copy the generated prompt
5. 🤖 Paste into ChatGPT/Claude/Copilot
6. 📊 Get your personalized weekly plan!

## 📁 Files

| File | Purpose | Update? |
|------|---------|---------|
| `Activities.csv` | 📊 Garmin export | ✅ Weekly |
| `my_training_goals.txt` | 🎯 Goals & constraints | 📝 As needed |
| `body_feedback.txt` | 💪 Weekly check-in | 🔄 Optional |
| `weekly_plan_YYYYMMDD.txt` | 📄 AI prompts | 🤖 Auto-saved |

## 🔍 What's Analyzed

- 📈 4-week training history & patterns
- ⚡ 7-day & 28-day training load (acute/chronic ratio)
- 🏃‍♂️ Recent ultra runs or high-intensity sessions
- 🏁 Race calendar & injury concerns
- 💭 Current body feedback (fatigue, soreness)

## 💡 Pro Tips

- ✅ Export **all** activities each week (not just new ones)
- 🎯 Be honest with body feedback - AI adjusts based on it
- 🔄 Update goals when races or priorities change
- 📚 Old plans are saved with dates for tracking

---

**Happy Training!** 🏃‍♂️ 🚴‍♂️ 🏊‍♂️ 💪
