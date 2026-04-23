"""
市場策略監控儀表板 v4.2 - Streamlit 版本
打開網頁即時抓取最新市場數據

用法：
  本機測試：
    pip install streamlit yfinance requests beautifulsoup4 pandas numpy
    streamlit run app.py
  
  雲端部署：
    上傳到 GitHub → 連結 Streamlit Cloud → 完成！
"""

import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# ========== 頁面設定 ==========
st.set_page_config(
    page_title="市場策略儀表板 v4.2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== 資料抓取函式 ==========

@st.cache_data(ttl=3600)  # 快取1小時
def fetch_vix():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="3mo")
        if hist.empty:
            return None, None
        current = float(hist["Close"].iloc[-1])
        ma_21 = float(hist["Close"].tail(21).mean())
        return current, ma_21
    except Exception as e:
        return None, None


@st.cache_data(ttl=3600)
def fetch_qqq_data():
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
        rsi_weekly = calc_rsi(weekly, 14)
        
        return {
            "price": current,
            "high_52w": high_52w,
            "drawdown": drawdown,
            "rsi_weekly": rsi_weekly
        }
    except Exception as e:
        return None


def calc_rsi(prices, period=14):
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


@st.cache_data(ttl=3600)
def fetch_breadth():
    try:
        rsp = yf.Ticker("RSP").history(period="1y")
        spy = yf.Ticker("SPY").history(period="1y")
        
        if rsp.empty or spy.empty or len(rsp) < 126 or len(spy) < 126:
            return None
        
        current_ratio = float(rsp["Close"].iloc[-1] / spy["Close"].iloc[-1])
        past_ratio = float(rsp["Close"].iloc[-126] / spy["Close"].iloc[-126])
        change_pct = (current_ratio - past_ratio) / past_ratio * 100
        
        return change_pct
    except Exception:
        return None


@st.cache_data(ttl=86400)  # CAPE 每天更新一次就夠
def fetch_cape():
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
    except Exception:
        return None


@st.cache_data(ttl=1800)  # 半小時
def fetch_fear_greed():
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
    except Exception:
        return None


# ========== 訊號計算 ==========

def calc_bottom_score(vix, fg_current, drawdown, rsi_weekly):
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
        breakdown.append((label, passed, pts))
    return score, breakdown


def calc_top_score(cape, fg_ma10, vix_ma21, breadth_change, rsi_weekly):
    breakdown = []
    
    cape_ok = cape is not None and cape >= 35
    breakdown.append(("CAPE >= 35 (啟動門檻)", cape_ok, 1, True))
    
    if not cape_ok:
        return 0, breakdown, False
    
    score = 1
    checks = [
        ("F&G 10日均 >= 75", fg_ma10 >= 75, 1),
        ("F&G 10日均 >= 85", fg_ma10 >= 85, 1),
        ("VIX 21日均 < 13", vix_ma21 is not None and vix_ma21 < 13, 1),
        ("RSP/SPY 6個月跌 > 5%", breadth_change is not None and breadth_change < -5, 2),
        ("週線 RSI(14) > 75", rsi_weekly is not None and rsi_weekly > 75, 1),
        ("週線 RSI(14) > 80", rsi_weekly is not None and rsi_weekly > 80, 1),
    ]
    for label, passed, pts in checks:
        if passed:
            score += pts
        breakdown.append((label, passed, pts, False))
    return score, breakdown, True


def calc_mid_signal(drawdown, fg_current, vix, rsi_weekly):
    cond_a = drawdown <= -10
    if not cond_a:
        return None
    
    cond_fg = fg_current <= 30
    cond_vix = vix > 25
    cond_rsi = rsi_weekly is not None and rsi_weekly < 45
    
    if not (cond_fg or cond_vix or cond_rsi):
        return None
    
    if drawdown <= -20 and fg_current <= 15:
        return {"level": "重度", "shift": 70}
    elif drawdown <= -15 and fg_current <= 20:
        return {"level": "中度", "shift": 50}
    else:
        return {"level": "輕度", "shift": 30}


def get_holding_zone(top_score):
    if top_score <= 2:
        return ("🟢 綠燈區", "健康多頭", "100% QQQ", "可持有TQQQ Runner")
    elif top_score <= 4:
        return ("🟡 黃燈區", "高檔警戒", "100% QQQ", "TQQQ Runner 需降落")
    elif top_score == 5:
        return ("🟠 紅燈區", "頂部成型", "50% QQQI + 50% QQQ", "防禦啟動")
    elif top_score == 6:
        return ("🔴 深紅燈區", "加強防備", "70% QQQI + 30% QQQ", "加強防守")
    else:
        return ("⚫ 極端頂部", "全面掩護", "100% QQQI", "全面掩護")


# ========== UI 主介面 ==========

st.title("📊 市場策略儀表板 v4.2")
st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')} · 三層訊號系統 + 風險遞進階梯")

# 重新整理按鈕
col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("🔄 重新整理數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 載入中提示
with st.spinner("📡 正在抓取最新市場數據..."):
    vix_current, vix_ma21 = fetch_vix()
    qqq_data = fetch_qqq_data()
    breadth = fetch_breadth()
    cape = fetch_cape()
    fg = fetch_fear_greed()

# 容錯
if vix_current is None:
    vix_current, vix_ma21 = 20, 20
if qqq_data is None:
    qqq_data = {"price": 0, "high_52w": 0, "drawdown": 0, "rsi_weekly": 50}
if cape is None:
    st.warning("⚠️ CAPE 抓取失敗，使用預設值 38")
    cape = 38.0
if fg is None:
    fg = {"current": 50, "rating": "Neutral", "ma_10": 50}
if breadth is None:
    breadth = 0

drawdown = qqq_data["drawdown"]
rsi_weekly = qqq_data["rsi_weekly"]

# 計算訊號
bottom_score, bottom_breakdown = calc_bottom_score(
    vix_current, fg["current"], drawdown, rsi_weekly
)
top_score, top_breakdown, top_active = calc_top_score(
    cape, fg["ma_10"], vix_ma21, breadth, rsi_weekly
)
mid_signal = calc_mid_signal(drawdown, fg["current"], vix_current, rsi_weekly)
zone_label, zone_state, zone_holding, zone_detail = get_holding_zone(top_score)

# ========== 主建議框 ==========

# 決定最終建議
if bottom_score >= 5:
    if bottom_score >= 7:
        action_title = "🚀 TQQQ 全壓（最後40%）"
    elif bottom_score >= 6:
        action_title = "🚀 TQQQ 加碼至60%"
    else:
        action_title = "🚀 TQQQ 首批30%"
    action_color = "error"
    action_layer = "🟢 底部訊號"
    action_detail = f"底部評分 {bottom_score}/9 - 分批買入TQQQ抓反彈"
    action_target = f"TQQQ {30 if bottom_score == 5 else 60 if bottom_score == 6 else 100}%"
elif mid_signal:
    action_title = f"🟡 QQQI 轉 {mid_signal['shift']}% 至 QQQ"
    action_color = "warning"
    action_layer = f"🟡 中度訊號 ({mid_signal['level']})"
    action_detail = f"回撤 {drawdown:.1f}% 觸發修正級調整"
    action_target = f"QQQ {mid_signal['shift']}% + QQQI {100-mid_signal['shift']}%"
else:
    action_title = f"持倉：{zone_holding}"
    action_color = "success" if top_score <= 2 else "warning" if top_score <= 4 else "error"
    action_layer = f"{zone_label}（頂部評分 {top_score}）"
    action_detail = zone_detail
    action_target = zone_holding

# 顯示主建議
if action_color == "success":
    st.success(f"**{action_layer}**\n\n### {action_title}\n\n{action_detail}\n\n🎯 **目標持倉：{action_target}**")
elif action_color == "warning":
    st.warning(f"**{action_layer}**\n\n### {action_title}\n\n{action_detail}\n\n🎯 **目標持倉：{action_target}**")
else:
    st.error(f"**{action_layer}**\n\n### {action_title}\n\n{action_detail}\n\n🎯 **目標持倉：{action_target}**")

# ========== 關鍵指標 ==========

st.subheader("📊 關鍵指標")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    cape_status = '低估' if cape < 20 else '合理' if cape < 28 else '偏高' if cape < 35 else '極度高估'
    st.metric("Shiller CAPE", f"{cape:.1f}", cape_status)

with col2:
    vix_status = '自滿' if vix_current < 15 else '正常' if vix_current < 20 else '緊張' if vix_current < 30 else '恐慌'
    st.metric("VIX", f"{vix_current:.1f}", f"21日均 {vix_ma21:.1f} · {vix_status}")

with col3:
    st.metric("Fear & Greed", f"{int(fg['current'])}", f"10日均 {fg['ma_10']:.0f} · {fg['rating']}")

with col4:
    dd_delta = f"-{abs(drawdown):.1f}%" if drawdown < 0 else f"{drawdown:.1f}%"
    st.metric("QQQ 回撤", f"{drawdown:.1f}%", dd_delta, delta_color="inverse")

with col5:
    rsi_display = f"{rsi_weekly:.1f}" if rsi_weekly else "N/A"
    rsi_status = "超賣" if rsi_weekly and rsi_weekly < 35 else "超買" if rsi_weekly and rsi_weekly > 75 else "正常"
    st.metric("QQQ 週線 RSI", rsi_display, rsi_status)

with col6:
    breadth_status = "惡化" if breadth < -5 else "健康"
    st.metric("RSP/SPY 6月變化", f"{breadth:+.1f}%", breadth_status)

# ========== 風險遞進階梯 ==========

st.subheader(f"🎯 風險遞進階梯（當前頂部評分：{top_score}/8）")

zones_data = [
    ("0-2", "🟢 綠燈", "100% QQQ + Runner", top_score <= 2),
    ("3-4", "🟡 黃燈", "100% QQQ (Runner降落)", 3 <= top_score <= 4),
    ("5", "🟠 紅燈", "50% QQQI + 50% QQQ", top_score == 5),
    ("6", "🔴 深紅", "70% QQQI + 30% QQQ", top_score == 6),
    ("7+", "⚫ 極端", "100% QQQI", top_score >= 7),
]

zone_cols = st.columns(5)
for i, (score_range, label, holding, is_active) in enumerate(zones_data):
    with zone_cols[i]:
        if is_active:
            st.info(f"**{label}**\n\n{score_range} 分\n\n{holding}")
        else:
            st.caption(f"{label}\n\n{score_range} 分\n\n{holding}")

# ========== 三層訊號細項 ==========

st.subheader("📋 訊號評分細項")

tab1, tab2, tab3 = st.tabs(["🟢 底部訊號", "🔴 頂部訊號", "🟡 中度訊號"])

with tab1:
    st.markdown(f"**底部訊號評分：{bottom_score}/9**")
    st.caption("達5分觸發 TQQQ 分批進場" if bottom_score < 5 else "🚨 已達5分觸發 TQQQ")
    
    for label, passed, pts in bottom_breakdown:
        mark = "✅" if passed else "⬜"
        st.write(f"{mark} {label} (+{pts})")

with tab2:
    st.markdown(f"**頂部訊號評分：{top_score}/8**")
    if top_active:
        st.caption("🛡️ CAPE門檻已達，開始評分")
    else:
        st.caption(f"CAPE {cape:.1f} < 35，系統休眠")
    
    for item in top_breakdown:
        if len(item) == 4:
            label, passed, pts, is_threshold = item
            marker = " 🔒" if is_threshold else ""
        else:
            label, passed, pts = item
            marker = ""
        mark = "✅" if passed else "⬜"
        st.write(f"{mark} {label}{marker} (+{pts})")

with tab3:
    if mid_signal:
        st.success(f"**中度訊號：{mid_signal['level']}級**")
        st.write(f"- QQQ 回撤達 {drawdown:.1f}%")
        st.write(f"- 建議：{mid_signal['shift']}% QQQI → QQQ")
    else:
        st.info("**中度訊號：未觸發**")
        st.write(f"- QQQ 回撤 {drawdown:.1f}% 未達 -10% 門檻，或輔助條件未滿足")

# ========== 頁尾 ==========

st.divider()
st.caption("📖 SOP v4.2 · 三層訊號系統 + 風險遞進階梯")
st.caption("⚠️ 本儀表板僅供參考，不構成投資建議")
