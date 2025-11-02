"""
Garmin Training Data Processor
This script processes Garmin training data and prepares it for AI-powered training planning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

class GarminDataProcessor:
    """Process and analyze Garmin training data for AI-based training planning."""
    
    def __init__(self, csv_file_path):
        """Initialize with path to Garmin CSV export."""
        self.csv_file = csv_file_path
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Load and parse the CSV file."""
        # Load the CSV with Danish column names
        self.df = pd.read_csv(self.csv_file)
        
        # Convert date column to datetime
        self.df['Dato'] = pd.to_datetime(self.df['Dato'])
        
        # Sort by date descending (most recent first)
        self.df = self.df.sort_values('Dato', ascending=False).reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df)} activities")
        print(f"  Date range: {self.df['Dato'].min().date()} to {self.df['Dato'].max().date()}")
        return self
    
    def clean_data(self):
        """Clean and standardize the data."""
        self.df_clean = self.df.copy()
        
        # Clean numeric columns (remove commas for Danish decimal format)
        numeric_cols = ['Distance', 'Kalorier', 'Gennemsnitlig puls', 'Maks. puls', 
                       'Aerob TE', 'Samlet stigning', 'Skridt']
        
        for col in numeric_cols:
            if col in self.df_clean.columns:
                # Replace comma with dot for decimal numbers
                self.df_clean[col] = self.df_clean[col].astype(str).str.replace(',', '.')
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
        
        # Parse time duration (format: H:MM or H:MM:SS)
        self.df_clean['Duration_Minutes'] = self.df_clean['Tid'].apply(self._parse_duration)
        
        print(f"✓ Cleaned data")
        return self
    
    def _parse_duration(self, time_str):
        """Convert time string (H:MM or H:MM:SS) to minutes."""
        try:
            if pd.isna(time_str):
                return np.nan
            parts = str(time_str).split(':')
            if len(parts) == 2:  # H:MM
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # H:MM:SS
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        except:
            return np.nan
    
    def get_weekly_summary(self, weeks=4):
        """Get weekly training summary for the last N weeks."""
        df = self.df_clean.copy()
        df['Week'] = df['Dato'].dt.isocalendar().week
        df['Year'] = df['Dato'].dt.year
        
        # Filter to last N weeks
        latest_date = df['Dato'].max()
        cutoff_date = latest_date - timedelta(weeks=weeks)
        df_recent = df[df['Dato'] >= cutoff_date]
        
        # Group by week and activity type
        weekly = df_recent.groupby(['Year', 'Week', 'Aktivitetstype']).agg({
            'Distance': 'sum',
            'Duration_Minutes': 'sum',
            'Kalorier': 'sum',
            'Aerob TE': 'mean',
            'Dato': 'count'
        }).round(2)
        
        weekly.columns = ['Total_Distance_km', 'Total_Duration_min', 
                         'Total_Calories', 'Avg_Aerobic_TE', 'Activity_Count']
        
        return weekly.reset_index()
    
    def get_activity_distribution(self):
        """Get distribution of activity types."""
        return self.df_clean.groupby('Aktivitetstype').agg({
            'Dato': 'count',
            'Distance': 'sum',
            'Duration_Minutes': 'sum',
            'Kalorier': 'sum'
        }).round(2).rename(columns={
            'Dato': 'Count',
            'Distance': 'Total_Distance_km',
            'Duration_Minutes': 'Total_Duration_min',
            'Kalorier': 'Total_Calories'
        })
    
    def get_training_load_trend(self, weeks=8):
        """Calculate weekly training load trend."""
        df = self.df_clean.copy()
        latest_date = df['Dato'].max()
        cutoff_date = latest_date - timedelta(weeks=weeks)
        df_recent = df[df['Dato'] >= cutoff_date].copy()
        
        # Group by week
        df_recent.loc[:, 'Week_Start'] = df_recent['Dato'].dt.to_period('W').apply(lambda r: r.start_time)
        
        weekly_load = df_recent.groupby('Week_Start').agg({
            'Duration_Minutes': 'sum',
            'Distance': 'sum',
            'Aerob TE': 'sum',
            'Kalorier': 'sum'
        }).round(2)
        
        weekly_load.columns = ['Total_Minutes', 'Total_Distance_km', 
                               'Total_Training_Effect', 'Total_Calories']
        
        # Reset index and convert to string dates for JSON compatibility
        weekly_load = weekly_load.reset_index()
        weekly_load['Week_Start'] = weekly_load['Week_Start'].astype(str)
        weekly_load = weekly_load.set_index('Week_Start')
        
        return weekly_load
    
    def get_personal_records(self):
        """Extract personal records and key metrics."""
        df = self.df_clean
        
        records = {
            'longest_run_km': df[df['Aktivitetstype'].str.contains('Løb', na=False)]['Distance'].max(),
            'longest_run_date': df[df['Aktivitetstype'].str.contains('Løb', na=False)].loc[
                df[df['Aktivitetstype'].str.contains('Løb', na=False)]['Distance'].idxmax(), 'Dato'
            ].date() if not df[df['Aktivitetstype'].str.contains('Løb', na=False)].empty else None,
            
            'longest_duration_min': df['Duration_Minutes'].max(),
            'total_activities': len(df),
            'total_distance_km': df['Distance'].sum(),
            'total_duration_hours': df['Duration_Minutes'].sum() / 60,
            'avg_weekly_distance_km': None,
            'avg_weekly_duration_hours': None,
        }
        
        # Calculate weekly averages using actual weekly grouping (more accurate)
        df_copy = df.copy()
        df_copy['Week_Start'] = df_copy['Dato'].dt.to_period('W').apply(lambda r: r.start_time)
        
        weekly_totals = df_copy.groupby('Week_Start').agg({
            'Distance': 'sum',
            'Duration_Minutes': 'sum'
        })
        
        if len(weekly_totals) > 0:
            records['avg_weekly_distance_km'] = round(weekly_totals['Distance'].mean(), 1)
            records['avg_weekly_duration_hours'] = round((weekly_totals['Duration_Minutes'].mean() / 60), 1)
        
        return records
    
    def prepare_ai_context(self, weeks_history=4):
        """Prepare comprehensive training context for AI chat."""
        
        context = {
            'summary': {
                'date_generated': datetime.now().strftime('%Y-%m-%d'),
                'data_period': f"Last {weeks_history} weeks",
            },
            'personal_records': self.get_personal_records(),
            'activity_distribution': self.get_activity_distribution().to_dict(),
            'weekly_summary': self.get_weekly_summary(weeks=weeks_history).to_dict(),
            'training_load_trend': self.get_training_load_trend(weeks=weeks_history).to_dict(),
            'recent_activities': self.df_clean.head(10)[
                ['Dato', 'Aktivitetstype', 'Distance', 'Duration_Minutes', 
                 'Gennemsnitlig puls', 'Aerob TE']
            ].to_dict(orient='records')
        }
        
        return context
    
    def get_significant_events(self, days_back=21, ultra_threshold_km=42):
        """Identify significant training events (ultra runs, very high load) from recent weeks.
        
        Args:
            days_back: How many days back to check (default 21 = 3 weeks)
            ultra_threshold_km: Distance threshold for "ultra" runs (default 42km)
        
        Returns:
            DataFrame with significant events
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df_recent = self.df_clean[self.df_clean['Dato'] >= cutoff_date].copy()
        
        # Find ultra-distance runs or very high training load
        significant = df_recent[
            ((df_recent['Aktivitetstype'].str.contains('Løb', na=False)) & 
             (df_recent['Distance'] >= ultra_threshold_km)) |
            (df_recent['Aerob TE'] >= 4.5)
        ].sort_values('Dato', ascending=False)
        
        return significant
    
    def get_training_load_summary(self):
        """Calculate acute (7-day) and chronic (28-day) training load metrics.
        
        Returns:
            Dictionary with load metrics
        """
        now = datetime.now()
        
        # 7-day load (acute)
        cutoff_7d = now - timedelta(days=7)
        df_7d = self.df_clean[self.df_clean['Dato'] >= cutoff_7d].copy()
        
        # 28-day load (chronic)
        cutoff_28d = now - timedelta(days=28)
        df_28d = self.df_clean[self.df_clean['Dato'] >= cutoff_28d].copy()
        
        # Calculate metrics
        load_7d = df_7d['Aerob TE'].sum() if not df_7d.empty else 0
        avg_te_7d = df_7d['Aerob TE'].mean() if not df_7d.empty else 0
        avg_te_28d = df_28d['Aerob TE'].mean() if not df_28d.empty else 0
        load_28d = df_28d['Aerob TE'].sum() if not df_28d.empty else 0
        
        # Acute:Chronic ratio (7-day vs 28-day average load per day)
        avg_daily_7d = load_7d / 7 if load_7d > 0 else 0
        avg_daily_28d = load_28d / 28 if load_28d > 0 else 0
        ac_ratio = avg_daily_7d / avg_daily_28d if avg_daily_28d > 0 else 0
        
        return {
            'load_7d': round(load_7d, 1),
            'load_28d': round(load_28d, 1),
            'avg_te_7d': round(avg_te_7d, 2),
            'avg_te_28d': round(avg_te_28d, 2),
            'ac_ratio': round(ac_ratio, 2),
            'activities_7d': len(df_7d),
            'activities_28d': len(df_28d)
        }
    
    def get_race_warnings(self, goal_text):
        """Parse race dates from goals and generate proximity warnings.
        
        Args:
            goal_text: Training goals text containing race calendar
            
        Returns:
            List of warning strings for upcoming races
        """
        import re
        warnings = []
        
        if not goal_text:
            return warnings
        
        # Look for dates in format: | Dec 7, 2025 | Event | Distance |
        race_pattern = r'\|\s*([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
        matches = re.finditer(race_pattern, goal_text)
        
        now = datetime.now()
        
        for match in matches:
            try:
                date_str = match.group(1).strip()
                event_name = match.group(2).strip()
                distance = match.group(3).strip()
                
                race_date = datetime.strptime(date_str, '%b %d, %Y')
                days_until = (race_date - now).days
                weeks_until = round(days_until / 7)
                
                if days_until < 0:
                    continue  # Skip past races
                
                # Generate warnings for races within next 8 weeks
                if days_until <= 7:
                    warnings.append(f"🔴 **RACE THIS WEEK:** {event_name} ({distance}) on {date_str} - TAPER MODE")
                elif days_until <= 14:
                    warnings.append(f"🟠 **RACE IN 2 WEEKS:** {event_name} ({distance}) on {date_str} - Begin taper")
                elif days_until <= 28:
                    warnings.append(f"🟡 **RACE IN {weeks_until} WEEKS:** {event_name} ({distance}) on {date_str} - Peak phase")
                elif days_until <= 56:
                    warnings.append(f"🟢 **RACE IN {weeks_until} WEEKS:** {event_name} ({distance}) on {date_str} - Build phase")
                    
            except Exception:
                continue  # Skip if date parsing fails
        
        return warnings
    
    def generate_ai_prompt(self, goal=None, weeks_history=4):
        """Generate a prompt for AI training planning."""
        
        context = self.prepare_ai_context(weeks_history=weeks_history)
        
        # Build the prompt
        prompt = f"""# Training Planning Context

## Personal Records & Overall Stats
- Total Activities: {context['personal_records']['total_activities']}
- Total Distance: {context['personal_records']['total_distance_km']:.1f} km
- Total Training Time: {context['personal_records']['total_duration_hours']:.1f} hours
- Longest Run: {context['personal_records']['longest_run_km']:.1f} km
- Average Weekly Distance: {context['personal_records']['avg_weekly_distance_km']:.1f} km
- Average Weekly Duration: {context['personal_records']['avg_weekly_duration_hours']:.1f} hours

## Training Load Summary
"""
        
        # Add training load metrics
        load_summary = self.get_training_load_summary()
        prompt += f"""- **7-Day Load:** {load_summary['load_7d']} ({load_summary['activities_7d']} activities, Avg TE: {load_summary['avg_te_7d']})
- **28-Day Load:** {load_summary['load_28d']} ({load_summary['activities_28d']} activities, Avg TE: {load_summary['avg_te_28d']})
- **Acute:Chronic Ratio:** {load_summary['ac_ratio']} (Optimal: 0.8-1.3, >1.5 = injury risk)
"""

        # Add A/C ratio warnings
        if load_summary['ac_ratio'] > 1.5:
            prompt += "\n🔴 **WARNING:** Acute:Chronic ratio >1.5 indicates HIGH INJURY RISK. Consider reducing volume.\n"
        elif load_summary['ac_ratio'] > 1.3:
            prompt += "\n🟠 **CAUTION:** Acute:Chronic ratio >1.3 indicates fatigue accumulation. Monitor carefully.\n"
        elif load_summary['ac_ratio'] < 0.8:
            prompt += "\n🟢 **OPPORTUNITY:** Low A/C ratio - safe to ramp up training volume if feeling good.\n"
        
        prompt += "\n💡 *Load Guidance: Ratio <0.8 = ramping up safely possible, 0.8-1.3 = optimal training zone, >1.3 = fatigue accumulating*\n"
        
        # Add race proximity warnings
        if goal:
            race_warnings = self.get_race_warnings(goal)
            if race_warnings:
                prompt += "\n## 🏁 Upcoming Race Alerts\n\n"
                for warning in race_warnings:
                    prompt += f"{warning}\n"
                prompt += "\n"

        prompt += """
## Recent Significant Events (Last 3 Weeks)
"""
        
        # Add significant events section
        significant_events = self.get_significant_events(days_back=21)
        if not significant_events.empty:
            prompt += "\n⚠️ **High-impact training sessions requiring recovery consideration:**\n"
            for _, event in significant_events.iterrows():
                date = event['Dato'].strftime('%Y-%m-%d')
                days_ago = (datetime.now() - event['Dato']).days
                prompt += f"\n- **{date}** ({days_ago} days ago) - {event['Aktivitetstype']}: "
                prompt += f"{event['Distance']:.1f} km, {event['Duration_Minutes']:.0f} min"
                if not pd.isna(event['Aerob TE']):
                    prompt += f", TE: {event['Aerob TE']:.1f}"
                if not pd.isna(event.get('Samlet stigning')) and event['Samlet stigning'] > 0:
                    prompt += f", Elevation: {event['Samlet stigning']:.0f}m"
                prompt += "\n"
        else:
            prompt += "\nNo ultra-distance or very high intensity sessions in the last 3 weeks.\n"
        
        prompt += f"""
## Recent Training Pattern (Last {weeks_history} weeks)
"""
        
        # Add weekly summary
        weekly_summary = self.get_weekly_summary(weeks=weeks_history)
        if not weekly_summary.empty:
            for _, row in weekly_summary.iterrows():
                prompt += f"\n**Week {row['Week']}/{row['Year']} - {row['Aktivitetstype']}:**\n"
                prompt += f"  - Activities: {int(row['Activity_Count'])}\n"
                prompt += f"  - Distance: {row['Total_Distance_km']:.1f} km\n"
                prompt += f"  - Duration: {row['Total_Duration_min']:.0f} min\n"
                prompt += f"  - Avg Aerobic TE: {row['Avg_Aerobic_TE']:.1f}\n"
        
        # Add recent activities
        prompt += "\n## Most Recent Activities (Last 10)\n"
        for activity in context['recent_activities']:
            date = pd.to_datetime(activity['Dato']).strftime('%Y-%m-%d')
            prompt += f"\n- **{date}** - {activity['Aktivitetstype']}: "
            prompt += f"{activity['Distance']:.1f} km, {activity['Duration_Minutes']:.0f} min"
            if not pd.isna(activity['Aerob TE']):
                prompt += f", TE: {activity['Aerob TE']:.1f}"
            prompt += "\n"
        
        # Add goal if provided
        if goal:
            prompt += f"\n## Training Goal\n{goal}\n"
        
        # Add optional body feedback if file exists
        try:
            with open('body_feedback.txt', 'r', encoding='utf-8') as f:
                body_feedback = f.read().strip()
                if body_feedback:
                    prompt += f"\n## Body Feedback\n{body_feedback}\n"
        except FileNotFoundError:
            pass  # Optional file, skip if not present
        
        # Add dynamic week context based on current time
        now = datetime.now()
        today_name = now.strftime('%A')
        today_date = now.strftime('%B %d, %Y')
        
        # Calculate the week range for planning
        # If it's Sunday-Tuesday morning, include today in the planning week
        # Otherwise, start planning from tomorrow
        if now.weekday() <= 1:  # 0=Monday, 1=Tuesday, 6=Sunday
            week_start_date = now
            week_start_day = today_name
        else:
            # Start from next day
            week_start_date = now + timedelta(days=1)
            week_start_day = week_start_date.strftime('%A')
        
        # Calculate week end (always next Sunday)
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.weekday() != 6:  # If today is not Sunday
            days_until_sunday = 7
        week_end_date = now + timedelta(days=days_until_sunday)
        week_end_formatted = week_end_date.strftime('%B %d')
        
        prompt += f"""
## Request
Based on my training history above, please:
1. Analyze my current training patterns and identify strengths/weaknesses
2. Suggest a balanced weekly training plan for the next week
3. Include specific workouts with recommended duration, intensity, and type
4. Consider recovery needs based on my recent training load
5. Help me progress toward my goals while avoiding overtraining

## This Week's Context
Today is {today_name}, {today_date}.
Next week starts on {week_start_day}, {week_start_date.strftime('%B %d, %Y')}.

Please provide a SPECIFIC weekly training plan for next week (week starting {week_start_date.strftime('%B %d')}):
1. List each day ({week_start_day}-Sunday) with specific workouts
2. Include distance/duration targets for each session
3. Specify intensity levels (easy, moderate, tempo, intervals)
4. Indicate which are key workouts vs. recovery sessions
5. Suggest alternatives if I need to adjust due to schedule conflicts

Format each day clearly so I can follow it easily.

**IMPORTANT: After creating the weekly plan, automatically generate a clean, formatted PDF version of the training plan that I can print or save. The PDF should include the week dates, daily workouts, and key notes/warnings.**
"""
        
        return prompt
    
    def export_for_ai(self, output_file='training_data_for_ai.json', weeks_history=4):
        """Export processed data to JSON for AI consumption."""
        context = self.prepare_ai_context(weeks_history=weeks_history)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✓ Exported training context to {output_file}")
        return output_file


def initial_setup():
    """Run initial setup to create baseline data and save training goals."""
    print("=" * 70)
    print("INITIAL SETUP - AI Training Planner")
    print("=" * 70)
    print()
    
    # Initialize processor
    processor = GarminDataProcessor('Activities.csv')
    processor.load_data()
    processor.clean_data()
    
    print("\n" + "=" * 70)
    print("YOUR TRAINING HISTORY")
    print("=" * 70)
    print(processor.get_activity_distribution())
    
    print("\n" + "=" * 70)
    print("PERSONAL RECORDS")
    print("=" * 70)
    records = processor.get_personal_records()
    for key, value in records.items():
        print(f"{key}: {value}")
    
    # Get training goals from user
    print("\n" + "=" * 70)
    print("DEFINE YOUR TRAINING GOALS")
    print("=" * 70)
    print("These goals will be saved and used for future weekly plans.")
    print()
    
    goals = input("Enter your training goals: ").strip()
    if not goals:
        goals = "Maintain balanced fitness across running, cycling, and swimming."
    
    # Save goals to file
    with open('my_training_goals.txt', 'w', encoding='utf-8') as f:
        f.write(goals)
    
    print(f"\n✓ Goals saved to 'my_training_goals.txt'")
    
    # Create initial baseline
    processor.export_for_ai(output_file='training_baseline.json', weeks_history=8)
    
    print("\n" + "=" * 70)
    print("INITIAL SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review your goals in 'my_training_goals.txt' (edit if needed)")
    print("2. Run 'python garmin_data_processor.py weekly' to get this week's plan")
    print("3. After training, export new data from Garmin and replace Activities.csv")
    print("4. Run weekly command again to get next week's plan")
    print()


def weekly_plan():
    """Generate weekly training plan based on recent data."""
    print("=" * 70)
    print("WEEKLY TRAINING PLAN GENERATOR")
    print("=" * 70)
    print()
    
    # Initialize processor
    processor = GarminDataProcessor('Activities.csv')
    processor.load_data()
    processor.clean_data()
    
    # Load saved goals
    try:
        with open('my_training_goals.txt', 'r', encoding='utf-8') as f:
            goals = f.read().strip()
    except FileNotFoundError:
        print("⚠ No training goals found. Running initial setup first...")
        print()
        initial_setup()
        return
    
    # Show last week's summary
    print("LAST WEEK'S TRAINING:")
    print("=" * 70)
    last_week = processor.get_weekly_summary(weeks=1)
    if not last_week.empty:
        for _, row in last_week.iterrows():
            print(f"{row['Aktivitetstype']:.<25} {int(row['Activity_Count'])} sessions, "
                  f"{row['Total_Distance_km']:.1f} km, {row['Total_Duration_min']:.0f} min")
    else:
        print("No activities found in the last week.")
    
    print()
    
    # Show last 4 weeks trend
    print("LAST 4 WEEKS TREND:")
    print("=" * 70)
    recent = processor.get_weekly_summary(weeks=4)
    weekly_totals = recent.groupby(['Year', 'Week']).agg({
        'Total_Distance_km': 'sum',
        'Total_Duration_min': 'sum',
        'Activity_Count': 'sum'
    }).round(1)
    
    for (year, week), row in weekly_totals.iterrows():
        print(f"Week {week}/{year}: {row['Activity_Count']:.0f} activities, "
              f"{row['Total_Distance_km']:.1f} km, {row['Total_Duration_min']:.0f} min")
    
    print()
    
    # Show training load warnings
    load_summary = processor.get_training_load_summary()
    print("TRAINING LOAD ANALYSIS:")
    print("=" * 70)
    print(f"7-Day Load: {load_summary['load_7d']} | 28-Day Load: {load_summary['load_28d']}")
    print(f"A/C Ratio: {load_summary['ac_ratio']}")
    
    if load_summary['ac_ratio'] > 1.5:
        print("🔴 WARNING: HIGH INJURY RISK - Consider reducing volume")
    elif load_summary['ac_ratio'] > 1.3:
        print("🟠 CAUTION: Fatigue accumulating - monitor carefully")
    elif load_summary['ac_ratio'] < 0.8:
        print("🟢 OPPORTUNITY: Safe to ramp up if feeling good")
    else:
        print("✅ Optimal training zone")
    print()
    
    # Show race proximity warnings
    race_warnings = processor.get_race_warnings(goals)
    if race_warnings:
        print("UPCOMING RACE ALERTS:")
        print("=" * 70)
        for warning in race_warnings:
            print(warning)
        print()
    
    # Generate prompt for next week
    prompt = processor.generate_ai_prompt(goal=goals, weeks_history=4)
    
    # Save prompt
    filename = f'weekly_plan_{datetime.now().strftime("%Y%m%d")}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    # Try to copy to clipboard
    if CLIPBOARD_AVAILABLE:
        try:
            pyperclip.copy(prompt)
            clipboard_msg = "✅ Prompt copied to clipboard!"
        except Exception:
            clipboard_msg = "⚠️ Clipboard copy failed (but file saved)"
    else:
        clipboard_msg = "ℹ️ Install pyperclip for auto-clipboard: pip install pyperclip"
    
    print("=" * 70)
    print("PROMPT GENERATED FOR AI CHAT")
    print("=" * 70)
    print(f"✓ Saved to '{filename}'")
    print(clipboard_msg)
    print()
    
    if CLIPBOARD_AVAILABLE:
        print("🎯 READY TO PASTE!")
        print("=" * 70)
        print("Just open ChatGPT/Claude/Copilot and press Ctrl+V")
    else:
        print("COPY THIS TO YOUR AI CHAT:")
        print("=" * 70)
        print()
        print(prompt)
    
    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    if CLIPBOARD_AVAILABLE:
        print("1. Open ChatGPT, Claude, or GitHub Copilot Chat")
        print("2. Press Ctrl+V to paste the prompt")
    else:
        print("1. Copy the prompt above")
        print("2. Paste into ChatGPT, Claude, or GitHub Copilot Chat")
    print("3. Receive your personalized weekly training plan")
    print("4. Save the AI's response for reference during the week")
    print()
    print("After completing the week:")
    print("1. Export updated Activities.csv from Garmin")
    print("2. Run: python garmin_data_processor.py weekly")
    print()


def interactive_body_feedback():
    """Interactively collect body feedback from user."""
    print("=" * 70)
    print("BODY FEEDBACK UPDATE")
    print("=" * 70)
    print()
    print("Quick check-in on how you're feeling (Scale 1-10, where 10 = best)")
    print()
    
    try:
        fatigue = input("Fatigue level (1-10, 10=fresh): ").strip() or "7"
        sleep = input("Sleep quality (1-10, 10=excellent): ").strip() or "7"
        motivation = input("Motivation (1-10, 10=very high): ").strip() or "7"
        ankle = input("Ankle soreness (1-10, 10=no pain): ").strip() or "10"
        knee = input("Knee soreness (1-10, 10=no pain): ").strip() or "10"
        energy = input("Energy level (low/normal/high): ").strip() or "normal"
        stress = input("Stress level (low/medium/high): ").strip() or "low"
        
        print()
        print("--- Schedule Preferences ---")
        commute_day = input("Preferred bike commute day this week (Mon/Tue/Wed/Thu/Fri or 'none'): ").strip() or "Thursday"
        
        print()
        notes = input("Any additional notes? ").strip() or ""
        
        # Convert to soreness scale (inverse for ankle/knee)
        ankle_soreness = 10 - int(ankle) if ankle.isdigit() else 0
        knee_soreness = 10 - int(knee) if knee.isdigit() else 0
        
        feedback = f"""## Body Feedback (Scale 1-10, where 10 = best/highest)

fatigue: {fatigue}
sleep_quality: {sleep}
motivation: {motivation}
ankle_soreness: {ankle_soreness}  # 0=no pain, 10=severe
knee_soreness: {knee_soreness}  # 0=no pain, 10=severe
energy_level: {energy}
stress_level: {stress}

## Schedule Preferences

preferred_bike_commute_day: {commute_day}  # Best day for 47km bike commute this week

notes: "{notes if notes else 'No additional notes.'}"
"""
        
        with open('body_feedback.txt', 'w', encoding='utf-8') as f:
            f.write(feedback)
        
        print()
        print("✅ Body feedback saved to 'body_feedback.txt'")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️ Cancelled")
        return


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'setup':
            initial_setup()
        elif command == 'weekly':
            weekly_plan()
        elif command == 'feedback':
            interactive_body_feedback()
        else:
            print("Unknown command. Available commands:")
            print()
            print("Usage:")
            print("  python garmin_data_processor.py setup     # Initial setup")
            print("  python garmin_data_processor.py weekly    # Generate weekly plan")
            print("  python garmin_data_processor.py feedback  # Update body feedback")
    else:
        # Check if setup has been done
        if os.path.exists('my_training_goals.txt'):
            weekly_plan()
        else:
            initial_setup()


if __name__ == "__main__":
    main()
