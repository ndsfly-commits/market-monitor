"""
================================================================
每日市場指標自動監控腳本 v4.2 (終極定稿版)
================================================================

三層訊號系統 + 風險遞進階梯 + TQQQ部位狀態追蹤

抓取指標：
- Shiller CAPE Ratio
- VIX 即時 + 21日滾動均值
- QQQ 現價、52週高點、週線RSI(14)
- RSP/SPY 比值（6個月寬度對比）
- CNN Fear & Greed Index 即時 + 10日滾動均值

計算三層訊號：
1. 底部訊號 (9分制)：VIX + F&G + 回撤 + RSI
2. 頂部訊號 (8分制 + CAPE≥35門檻)：CAPE + F&G + VIX + 寬度 + RSI
3. 中度訊號：回撤≥10% AND (F&G≤30 OR VIX>25 OR RSI<45)

風險遞進階梯（依頂部評分）：
- 0-2分：100% QQQ
- 3-4分：100% QQQ（TQQQ Runner強制降落）
- 5分：50% QQQI + 50% QQQ
- 6分：70% QQQI + 30% QQQ
- 7+分：100% QQQI

用法：
  pip install yfinance requests beautifulsoup4 pandas pandas-ta numpy
  python daily_monitor_v4.py

輸出：
- daily_data_v4.json : 完整數據 + 90天歷史
- dashboard_v4.html  : 互動式儀表板
================================================================
"""

import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"缺少套件：{e}")
    print("請執行：pip install yfinance requests beautifulsoup4 pandas numpy")
    sys.exit(1)

# pandas-ta 可選（若未安裝則用手算RSI）
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


# ========== 數據抓取 ==========

def fetch_vix_series():
    """抓取 VIX 過去60日數據，用於計算即時值 + 21日滾動均值"""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="3mo")
        if hist.empty:
            return None, None
        current = float(hist["Close"].iloc[-1])
        ma_21 = float(hist["Close"].tail(21).mean())
        return current, ma_21
    except Exception as e:
        print(f"⚠️  VIX 抓取失敗: {e}")
        return None, None


def fetch_qqq_data():
    """抓取 QQQ 資料：現價、52週高、週線RSI(14)"""
    try:
        qqq = yf.Ticker("QQQ")
        daily = qqq.history(period="2y")
        if daily.empty:
            return None
        
        current = float(daily["Close"].iloc[-1])
        high_52w = float(daily["High"].tail(252).max())
        drawdown = (current - high_52w) / high_52w * 100
        
        # 週線 RSI(14)
        weekly = daily["Close"].resample("W").last().dropna()
        
        if HAS_PANDAS_TA:
            rsi_series = ta.rsi(weekly, length=14)
            rsi_weekly = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        else:
            rsi_weekly = calc_rsi_manual(weekly, 14)
        
        return {
            "price": current,
            "high_52w": high_52w,
            "drawdown": drawdown,
            "rsi_weekly": rsi_weekly
        }
    except Exception as e:
        print(f"⚠️  QQQ 抓取失敗: {e}")
        return None


def calc_rsi_manual(prices, period=14):
    """手算 RSI"""
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else None
    except Exception:
        return None


def fetch_rsp_spy_breadth():
    """抓取 RSP/SPY 比值，計算126個交易日前的對比"""
    try:
        rsp = yf.Ticker("RSP").history(period="1y")
        spy = yf.Ticker("SPY").history(period="1y")
        
        if rsp.empty or spy.empty or len(rsp) < 126 or len(spy) < 126:
            return None
        
        current_ratio = float(rsp["Close"].iloc[-1] / spy["Close"].iloc[-1])
        past_ratio = float(rsp["Close"].iloc[-126] / spy["Close"].iloc[-126])
        change_pct = (current_ratio - past_ratio) / past_ratio * 100
        
        return {
            "current_ratio": current_ratio,
            "past_ratio_126d": past_ratio,
            "change_pct": change_pct
        }
    except Exception as e:
        print(f"⚠️  RSP/SPY 寬度抓取失敗: {e}")
        return None


def fetch_cape():
    """從 multpl.com 抓取 Shiller CAPE"""
    try:
        url = "https://www.multpl.com/shiller-pe"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        value_div = soup.find("div", id="current")
        if value_div:
            text = value_div.get_text()
            for token in text.replace("\n", " ").split():
                try:
                    val = float(token.replace(",", ""))
                    if 5 < val < 100:
                        return val
                except ValueError:
                    continue
        return None
    except Exception as e:
        print(f"⚠️  CAPE 抓取失敗: {e}")
        return None


def fetch_fear_greed():
    """抓取 Fear & Greed 即時值 + 10日均值"""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        current = data.get("fear_and_greed", {}).get("score")
        rating = data.get("fear_and_greed", {}).get("rating")
        
        historical = data.get("fear_and_greed_historical", {}).get("data", [])
        if historical and len(historical) >= 10:
            recent_10 = historical[-10:]
            ma_10 = sum(item["y"] for item in recent_10) / len(recent_10)
        else:
            ma_10 = current
        
        return {
            "current": round(float(current), 1),
            "rating": rating,
            "ma_10": round(float(ma_10), 1)
        }
    except Exception as e:
        print(f"⚠️  Fear & Greed 抓取失敗: {e}")
        return None


# ========== 三層訊號計算 ==========

def calc_bottom_score(vix, fg_current, drawdown, rsi_weekly):
    """底部訊號評分 (滿分9分)"""
    score = 0
    breakdown = []
    
    checks = [
        ("VIX >= 40", vix >= 40, 2),
        ("VIX >= 50", vix >= 50, 1),
        ("F&G <= 25", fg_current <= 25, 1),
        ("F&G <= 15", fg_current <= 15, 1),
        ("QQQ 回撤 >= -15%", drawdown <= -15, 1),
        ("QQQ 回撤 >= -20%", drawdown <= -20, 1),
        ("QQQ 回撤 >= -30%", drawdown <= -30, 1),
        ("週線 RSI(14) < 35", rsi_weekly is not None and rsi_weekly < 35, 1),
    ]
    
    for label, passed, pts in checks:
        if passed:
            score += pts
        breakdown.append({"label": label, "passed": passed, "pts": pts})
    
    return score, breakdown


def calc_top_score(cape, fg_ma10, vix_ma21, breadth_change, rsi_weekly):
    """頂部訊號評分 (CAPE≥35門檻 + 8分制)"""
    breakdown = []
    
    cape_threshold_passed = cape is not None and cape >= 35
    breakdown.append({
        "label": "CAPE >= 35 (啟動門檻)",
        "passed": cape_threshold_passed,
        "pts": 1,
        "is_threshold": True
    })
    
    if not cape_threshold_passed:
        return 0, breakdown, False
    
    score = 1
    
    checks = [
        ("F&G 10日均 >= 75", fg_ma10 >= 75, 1),
        ("F&G 10日均 >= 85", fg_ma10 >= 85, 1),
        ("VIX 21日均 < 13", vix_ma21 is not None and vix_ma21 < 13, 1),
        ("RSP/SPY 6個月跌 > 5%", 
         breadth_change is not None and breadth_change < -5, 2),
        ("週線 RSI(14) > 75", rsi_weekly is not None and rsi_weekly > 75, 1),
        ("週線 RSI(14) > 80", rsi_weekly is not None and rsi_weekly > 80, 1),
    ]
    
    for label, passed, pts in checks:
        if passed:
            score += pts
        breakdown.append({"label": label, "passed": passed, "pts": pts, "is_threshold": False})
    
    return score, breakdown, True


def calc_mid_signal(drawdown, fg_current, vix, rsi_weekly):
    """中度訊號判定"""
    cond_a = drawdown <= -10
    
    if not cond_a:
        return {
            "triggered": False,
            "level": "未觸發",
            "shift_pct": 0,
            "reason": f"QQQ回撤 {drawdown:.1f}% 未達 -10% 門檻"
        }
    
    cond_fg = fg_current <= 30
    cond_vix = vix > 25
    cond_rsi = rsi_weekly is not None and rsi_weekly < 45
    
    cond_b = cond_fg or cond_vix or cond_rsi
    
    if not cond_b:
        rsi_str = f"{rsi_weekly:.1f}" if rsi_weekly is not None else "N/A"
        return {
            "triggered": False,
            "level": "未觸發",
            "shift_pct": 0,
            "reason": f"回撤達標但輔助條件未達（F&G={fg_current}, VIX={vix:.1f}, RSI={rsi_str}）"
        }
    
    if drawdown <= -20 and fg_current <= 15:
        level = "重度"
        shift = 70
    elif drawdown <= -15 and fg_current <= 20:
        level = "中度"
        shift = 50
    else:
        level = "輕度"
        shift = 30
    
    reasons = []
    if cond_fg:
        reasons.append(f"F&G={fg_current:.0f}")
    if cond_vix:
        reasons.append(f"VIX={vix:.1f}")
    if cond_rsi:
        reasons.append(f"RSI週線={rsi_weekly:.1f}")
    
    return {
        "triggered": True,
        "level": level,
        "shift_pct": shift,
        "reason": f"回撤 {drawdown:.1f}% + " + " / ".join(reasons)
    }


# ========== 風險遞進階梯 ==========

def determine_holding_by_top_score(top_score):
    """依頂部評分決定持倉（v4.2 風險遞進階梯）"""
    if top_score <= 2:
        return {
            "zone": "🟢 綠燈區",
            "state": "健康多頭",
            "holding": "100% QQQ",
            "qqq_pct": 100,
            "qqqi_pct": 0,
            "tqqq_runner_ok": True,
            "detail": "可持有TQQQ Runner；QQQ抱緊"
        }
    elif top_score <= 4:
        return {
            "zone": "🟡 黃燈區",
            "state": "高檔警戒",
            "holding": "100% QQQ",
            "qqq_pct": 100,
            "qqqi_pct": 0,
            "tqqq_runner_ok": False,
            "detail": "強制TQQQ Runner降落轉QQQ，但QQQI防護罩尚未啟動"
        }
    elif top_score == 5:
        return {
            "zone": "🟠 紅燈區",
            "state": "頂部成型",
            "holding": "50% QQQI + 50% QQQ",
            "qqq_pct": 50,
            "qqqi_pct": 50,
            "tqqq_runner_ok": False,
            "detail": "防禦系統正式啟動，50%切換至QQQI"
        }
    elif top_score == 6:
        return {
            "zone": "🔴 深紅燈區",
            "state": "加強防備",
            "holding": "70% QQQI + 30% QQQ",
            "qqq_pct": 30,
            "qqqi_pct": 70,
            "tqqq_runner_ok": False,
            "detail": "70%切換至QQQI，加強防備"
        }
    else:
        return {
            "zone": "⚫ 極端頂部",
            "state": "全面掩護",
            "holding": "100% QQQI",
            "qqq_pct": 0,
            "qqqi_pct": 100,
            "tqqq_runner_ok": False,
            "detail": "全面QQQI掩護，等待底部訊號"
        }


def determine_final_action(bottom_score, mid_signal, top_score, holding_base):
    """綜合決策：底部 > 中度 > 頂部基準"""
    
    if bottom_score >= 5:
        if bottom_score >= 7:
            tqqq_pct = 100
            title = "🚀 TQQQ 全壓（最後40%）"
        elif bottom_score >= 6:
            tqqq_pct = 60
            title = "🚀 TQQQ 加碼至60%"
        else:
            tqqq_pct = 30
            title = "🚀 TQQQ 首批30%"
        
        return {
            "layer": "🟢 底部訊號",
            "urgency": "critical",
            "title": title,
            "detail": f"底部評分 {bottom_score}/9 - 分批買入TQQQ抓反彈",
            "target_holding": f"TQQQ {tqqq_pct}%",
            "color": "#a32d2d",
            "bg": "#fcebeb"
        }
    
    if mid_signal["triggered"]:
        return {
            "layer": f"🟡 中度訊號 ({mid_signal['level']})",
            "urgency": "medium",
            "title": f"QQQI 轉 {mid_signal['shift_pct']}% 至 QQQ",
            "detail": f"觸發原因：{mid_signal['reason']}",
            "target_holding": f"QQQ {mid_signal['shift_pct']}% + QQQI {100-mid_signal['shift_pct']}%（相對原本持倉）",
            "color": "#854f0b",
            "bg": "#faeeda"
        }
    
    urgency = "low" if top_score <= 2 else "medium" if top_score <= 4 else "high"
    color_map = {
        "low": ("#27500a", "#eaf3de"),
        "medium": ("#854f0b", "#faeeda"),
        "high": ("#a32d2d", "#fcebeb"),
    }
    color, bg = color_map[urgency]
    
    return {
        "layer": f"{holding_base['zone']}（頂部評分 {top_score}）",
        "urgency": urgency,
        "title": f"目標持倉：{holding_base['holding']}",
        "detail": holding_base["detail"],
        "target_holding": holding_base["holding"],
        "color": color,
        "bg": bg
    }


# ========== 主程式 ==========

def main():
    print("=" * 70)
    print(f"  每日市場監控 v4.2 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    print("\n📡 正在抓取市場數據...")
    
    vix_current, vix_ma21 = fetch_vix_series()
    qqq_data = fetch_qqq_data()
    breadth = fetch_rsp_spy_breadth()
    cape = fetch_cape()
    fg = fetch_fear_greed()
    
    if vix_current is None:
        vix_current, vix_ma21 = 20, 20
    if qqq_data is None:
        qqq_data = {"price": 0, "high_52w": 0, "drawdown": 0, "rsi_weekly": 50}
    if breadth is None:
        breadth = {"current_ratio": 0, "past_ratio_126d": 0, "change_pct": 0}
    if cape is None:
        print("⚠️  CAPE 抓取失敗，使用估計值 38")
        cape = 38.0
    if fg is None:
        fg = {"current": 50, "rating": "Neutral", "ma_10": 50}
    
    drawdown = qqq_data["drawdown"]
    rsi_weekly = qqq_data["rsi_weekly"]
    
    bottom_score, bottom_breakdown = calc_bottom_score(
        vix_current, fg["current"], drawdown, rsi_weekly
    )
    
    top_score, top_breakdown, top_active = calc_top_score(
        cape, fg["ma_10"], vix_ma21, breadth["change_pct"], rsi_weekly
    )
    
    mid_signal = calc_mid_signal(drawdown, fg["current"], vix_current, rsi_weekly)
    
    holding_base = determine_holding_by_top_score(top_score)
    
    action = determine_final_action(
        bottom_score, mid_signal, top_score, holding_base
    )
    
    print_terminal_report(vix_current, vix_ma21, qqq_data, breadth, cape, fg,
                          bottom_score, bottom_breakdown,
                          top_score, top_breakdown, top_active,
                          mid_signal, holding_base, action)
    
    today_data = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "raw": {
            "cape": cape,
            "vix_current": vix_current,
            "vix_ma21": vix_ma21,
            "fg_current": fg["current"],
            "fg_rating": fg["rating"],
            "fg_ma10": fg["ma_10"],
            "qqq_price": qqq_data["price"],
            "qqq_high_52w": qqq_data["high_52w"],
            "drawdown": drawdown,
            "rsi_weekly": rsi_weekly,
            "breadth_current": breadth["current_ratio"],
            "breadth_change_pct": breadth["change_pct"],
        },
        "bottom_score": bottom_score,
        "bottom_breakdown": bottom_breakdown,
        "top_score": top_score,
        "top_breakdown": top_breakdown,
        "top_active": top_active,
        "mid_signal": mid_signal,
        "holding_base": holding_base,
        "action": action,
    }
    
    save_data(today_data)
    generate_html(today_data)
    
    print(f"\n✅ 資料已儲存：daily_data_v4.json")
    print(f"🌐 儀表板已生成：dashboard_v4.html\n")


def print_terminal_report(vix_current, vix_ma21, qqq_data, breadth, cape, fg,
                          bottom_score, bottom_breakdown,
                          top_score, top_breakdown, top_active,
                          mid_signal, holding_base, action):
    """終端機報告"""
    rsi_str = f"{qqq_data['rsi_weekly']:.1f}" if qqq_data['rsi_weekly'] is not None else "N/A"
    
    print("\n📊 今日指標：")
    print(f"   CAPE             : {cape:.2f}")
    print(f"   VIX 即時 / 21日均 : {vix_current:.2f} / {vix_ma21:.2f}")
    print(f"   F&G 即時 / 10日均 : {fg['current']:.1f} ({fg['rating']}) / {fg['ma_10']:.1f}")
    print(f"   QQQ 現價 / 52週高 : ${qqq_data['price']:.2f} / ${qqq_data['high_52w']:.2f}")
    print(f"   QQQ 回撤         : {qqq_data['drawdown']:.2f}%")
    print(f"   QQQ 週線 RSI(14) : {rsi_str}")
    print(f"   RSP/SPY 6月變化  : {breadth['change_pct']:+.2f}%")
    
    print(f"\n🟢 底部訊號評分：{bottom_score}/9")
    for item in bottom_breakdown:
        mark = "✓" if item["passed"] else "○"
        print(f"   {mark} {item['label']} (+{item['pts']})")
    
    if top_active:
        print(f"\n🔴 頂部訊號評分：{top_score}/8 (CAPE門檻已達)")
    else:
        print(f"\n🔴 頂部訊號：系統休眠（CAPE {cape:.1f} < 35）")
    for item in top_breakdown:
        mark = "✓" if item["passed"] else "○"
        threshold_marker = " [門檻]" if item.get("is_threshold") else ""
        print(f"   {mark} {item['label']} (+{item['pts']}){threshold_marker}")
    
    print(f"\n🟡 中度訊號：{mid_signal['level']}")
    print(f"   {mid_signal['reason']}")
    if mid_signal["triggered"]:
        print(f"   建議：{mid_signal['shift_pct']}% QQQI → QQQ")
    
    print(f"\n📊 風險遞進階梯：{holding_base['zone']} - {holding_base['state']}")
    print(f"   基準持倉：{holding_base['holding']}")
    print(f"   TQQQ Runner 允許：{'是' if holding_base['tqqq_runner_ok'] else '否（須降落）'}")
    
    print(f"\n{'='*70}")
    print(f"💰 最終建議：{action['layer']}")
    print(f"   {action['title']}")
    print(f"   {action['detail']}")
    print(f"   目標持倉：{action['target_holding']}")
    print(f"{'='*70}")


def save_data(today_data):
    """儲存資料並更新90天歷史"""
    json_path = Path("daily_data_v4.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    else:
        full_data = {"latest": None, "history": []}
    
    full_data["latest"] = today_data
    
    history = full_data.get("history", [])
    history = [h for h in history if h["date"] != today_data["date"]]
    history.append(today_data)
    history = history[-90:]
    full_data["history"] = history
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)


def generate_html(today):
    """生成互動式HTML儀表板"""
    a = today["action"]
    raw = today["raw"]
    hb = today["holding_base"]
    
    rsi_display = f"{raw['rsi_weekly']:.1f}" if raw['rsi_weekly'] is not None else "N/A"
    rsi_zone = "正常"
    if raw['rsi_weekly'] is not None:
        if raw['rsi_weekly'] < 35:
            rsi_zone = "超賣"
        elif raw['rsi_weekly'] > 75:
            rsi_zone = "超買"
    
    # 底部訊號細項
    bottom_html = ""
    for item in today["bottom_breakdown"]:
        mark = "✓" if item["passed"] else "○"
        color = "#27500a" if item["passed"] else "#999"
        bottom_html += f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:0.5px solid #eee;font-size:12px;">
          <span style="color:{color};">{mark} {item['label']}</span>
          <span style="color:#999;">+{item['pts']}</span>
        </div>"""
    
    # 頂部訊號細項
    top_html = ""
    for item in today["top_breakdown"]:
        mark = "✓" if item["passed"] else "○"
        color = "#27500a" if item["passed"] else "#999"
        marker = " 🔒門檻" if item.get("is_threshold") else ""
        top_html += f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:0.5px solid #eee;font-size:12px;">
          <span style="color:{color};">{mark} {item['label']}{marker}</span>
          <span style="color:#999;">+{item['pts']}</span>
        </div>"""
    
    # 中度訊號
    ms = today["mid_signal"]
    if ms["triggered"]:
        ms_html = f"""
        <div style="padding:12px 14px;background:#faeeda;border-left:3px solid #854f0b;border-radius:6px;">
          <p style="margin:0;font-weight:500;color:#854f0b;">🟡 中度訊號：{ms['level']}</p>
          <p style="margin:6px 0 0;font-size:12px;color:#333;">轉換比例：{ms['shift_pct']}% QQQI → QQQ</p>
          <p style="margin:4px 0 0;font-size:11px;color:#888;">{ms['reason']}</p>
        </div>"""
    else:
        ms_html = f"""
        <div style="padding:12px 14px;background:#f5f5f0;border-radius:6px;">
          <p style="margin:0;color:#888;">🟡 中度訊號：未觸發</p>
          <p style="margin:4px 0 0;font-size:11px;color:#999;">{ms['reason']}</p>
        </div>"""
    
    # 風險遞進階梯
    zones_html = ""
    zones = [
        (0, 2, "🟢 綠燈", "100% QQQ + Runner", "#27500a", "#eaf3de"),
        (3, 4, "🟡 黃燈", "100% QQQ (Runner降落)", "#854f0b", "#faeeda"),
        (5, 5, "🟠 紅燈", "50% QQQI + 50% QQQ", "#cc5500", "#ffe8d6"),
        (6, 6, "🔴 深紅", "70% QQQI + 30% QQQ", "#a32d2d", "#fcebeb"),
        (7, 8, "⚫ 極端", "100% QQQI", "#4a1b0c", "#f0c0c0"),
    ]
    for low, high, label, holding, color, bg in zones:
        is_active = low <= today["top_score"] <= high
        border = f"2px solid {color}" if is_active else "0.5px solid #e0e0d8"
        range_text = f"{low}-{high}" if low != high else str(low)
        zones_html += f"""
        <div style="padding:10px;text-align:center;border-radius:8px;background:{bg if is_active else '#fafafa'};border:{border};">
          <p style="margin:0;font-size:12px;font-weight:500;color:{color if is_active else '#666'};">{label} ({range_text}分)</p>
          <p style="margin:4px 0 0;font-size:10px;color:{'#333' if is_active else '#999'};">{holding}</p>
        </div>"""
    
    dd_color = '#a32d2d' if raw['drawdown'] < -20 else '#854f0b' if raw['drawdown'] < -10 else '#222'
    breadth_color = '#a32d2d' if raw['breadth_change_pct'] < -5 else '#222'
    cape_status = '低估' if raw['cape'] < 20 else '合理' if raw['cape'] < 28 else '偏高' if raw['cape'] < 35 else '極度高估'
    vix_status = '自滿' if raw['vix_current'] < 15 else '正常' if raw['vix_current'] < 20 else '緊張' if raw['vix_current'] < 30 else '恐慌'
    
    bottom_status = '🚨 達5分觸發TQQQ' if today['bottom_score'] >= 5 else '需5分觸發TQQQ'
    top_status = '🛡️ 達5分觸發QQQI' if today['top_score'] >= 5 else f'CAPE門檻{"已達" if today["top_active"] else "未達"}，需5分觸發QQQI'
    
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>市場策略儀表板 v4.2 - {today['date']}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "PingFang TC", sans-serif; max-width: 1000px; margin: 20px auto; padding: 0 20px; background: #fafaf7; color: #222; line-height: 1.6; }}
  h1 {{ font-size: 26px; font-weight: 500; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .status-box {{ padding: 20px; border-radius: 12px; background: {a['bg']}; border-left: 5px solid {a['color']}; margin-bottom: 24px; }}
  .status-layer {{ display:inline-block; padding:3px 10px; border-radius:6px; font-size:11px; background:rgba(0,0,0,0.08); margin-bottom:8px; }}
  .status-title {{ font-size: 22px; font-weight: 500; color: {a['color']}; margin: 0 0 8px; }}
  .status-action {{ font-size: 14px; color: #333; margin: 0 0 6px; }}
  .status-target {{ font-size: 15px; color: {a['color']}; margin: 4px 0 0; font-weight: 500; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 24px; }}
  .metric {{ background: #fff; padding: 12px 14px; border-radius: 10px; border: 0.5px solid #e5e5e0; }}
  .metric-label {{ font-size: 11px; color: #888; margin: 0 0 4px; }}
  .metric-value {{ font-size: 20px; font-weight: 500; margin: 0; }}
  .metric-sub {{ font-size: 10px; color: #999; margin: 2px 0 0; }}
  .section {{ background: #fff; padding: 18px 20px; border-radius: 12px; border: 0.5px solid #e5e5e0; margin-bottom: 16px; }}
  .section h2 {{ font-size: 15px; font-weight: 500; margin: 0 0 12px; color: #333; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .footer {{ text-align: center; font-size: 11px; color: #999; margin: 30px 0 20px; }}
</style>
</head>
<body>

<h1>📊 市場策略儀表板 v4.2</h1>
<div class="subtitle">更新時間：{today['timestamp'][:19]} · 三層訊號系統 + 風險遞進階梯</div>

<div class="status-box">
  <span class="status-layer">{a['layer']}</span>
  <p class="status-title">{a['title']}</p>
  <p class="status-action">{a['detail']}</p>
  <p class="status-target">🎯 目標持倉：{a['target_holding']}</p>
</div>

<div class="metrics">
  <div class="metric">
    <p class="metric-label">Shiller CAPE</p>
    <p class="metric-value">{raw['cape']:.1f}</p>
    <p class="metric-sub">{cape_status}</p>
  </div>
  <div class="metric">
    <p class="metric-label">VIX (即時/21日均)</p>
    <p class="metric-value">{raw['vix_current']:.1f}</p>
    <p class="metric-sub">21日均 {raw['vix_ma21']:.1f} · {vix_status}</p>
  </div>
  <div class="metric">
    <p class="metric-label">F&G (即時/10日均)</p>
    <p class="metric-value">{int(raw['fg_current'])}</p>
    <p class="metric-sub">10日均 {raw['fg_ma10']:.0f} · {raw['fg_rating']}</p>
  </div>
  <div class="metric">
    <p class="metric-label">QQQ 回撤</p>
    <p class="metric-value" style="color:{dd_color}">{raw['drawdown']:.1f}%</p>
    <p class="metric-sub">${raw['qqq_price']:.2f} / ${raw['qqq_high_52w']:.2f}</p>
  </div>
  <div class="metric">
    <p class="metric-label">QQQ 週線 RSI</p>
    <p class="metric-value">{rsi_display}</p>
    <p class="metric-sub">{rsi_zone}</p>
  </div>
  <div class="metric">
    <p class="metric-label">RSP/SPY 寬度</p>
    <p class="metric-value" style="color:{breadth_color}">{raw['breadth_change_pct']:+.1f}%</p>
    <p class="metric-sub">6個月變化</p>
  </div>
</div>

<div class="section">
  <h2>🎯 風險遞進階梯（當前頂部評分：{today['top_score']}/8）</h2>
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px;">
    {zones_html}
  </div>
  <p style="margin:10px 0 0;font-size:11px;color:#888;">當前位置：{hb['zone']} · {hb['state']} · {hb['detail']}</p>
</div>

<div class="section">
  <h2>🟡 中度訊號（QQQI ↔ QQQ 微調）</h2>
  {ms_html}
</div>

<div class="two-col">
  <div class="section">
    <h2>🟢 底部訊號評分：{today['bottom_score']}/9</h2>
    <p style="margin:0 0 8px;font-size:11px;color:#888;">{bottom_status}</p>
    {bottom_html}
  </div>
  <div class="section">
    <h2>🔴 頂部訊號評分：{today['top_score']}/8</h2>
    <p style="margin:0 0 8px;font-size:11px;color:#888;">{top_status}</p>
    {top_html}
  </div>
</div>

<div class="footer">
  本儀表板由 daily_monitor_v4.py 自動生成 · 僅供參考，不構成投資建議<br>
  SOP v4.2 · 三層訊號系統 + 風險遞進階梯
</div>

</body>
</html>"""
    
    with open("dashboard_v4.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
