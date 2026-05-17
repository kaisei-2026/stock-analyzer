"""AI自動運用をバックグラウンドで定期実行するスケジューラー"""

import time
import sys
import os
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_agent_engine import run_ai_agent_cycle

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI Scheduler started.")
    
    while True:
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running AI cycle...")
            # force_run=False なので、内部の1時間リミッターに従う
            result = run_ai_agent_cycle(force_run=False)
            
            if result["ok"]:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cycle completed successfully.")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cycle skipped: {result.get('message')}")
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in scheduler: {str(e)}")
        
        # 5分ごとにチェック（内部のリミッターが1時間なので、実際には1時間おきに動く）
        time.sleep(300)

if __name__ == "__main__":
    main()
