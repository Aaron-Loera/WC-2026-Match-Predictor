"""CSS injection for the WC 2026 Predictor dashboard."""
import streamlit as st


def inject_theme() -> None:
    """Inject Google Fonts (Barlow Condensed) + full dark-theme CSS."""
    # Font link first so it loads in parallel
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed'
        ':wght@500;700;800&family=Barlow:wght@400;500&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* === STREAMLIT CHROME === */
        .block-container{padding-top:0!important;padding-left:3rem;padding-right:3rem;max-width:1500px}
        #MainMenu,footer{visibility:hidden}
        header[data-testid="stHeader"]{display:none!important}
        [data-testid="stDecoration"]{display:none!important}
        [data-testid="stMarkdownContainer"] p{margin:0}

        /* === ROOT PALETTE === */
        :root{
          --wc-bg:#1B1D1F;--wc-card:#26292C;--wc-bd:#34383C;--wc-track:#303438;
          --wc-txt:#E8EAEC;--wc-txt2:#9AA0A6;--wc-txt3:#7A8087;
          --wc-green:#7FB83E;--wc-gtxt:#9FD45B;--wc-gtint:#2A3A18;--wc-gline:#3A5A22;
          --wc-gold:#F2A93B;--wc-goldtxt:#F2B85C;--wc-grey:#80868C;--wc-red:#E5705B;
          --wc-r:10px;
        }

        /* === BANNER COMPONENT (full-bleed iframe) === */
        [data-testid="stCustomComponentV1"]{
          margin:0 -3rem!important;width:calc(100% + 6rem)!important;
          border-bottom:1px solid var(--wc-bd);
        }

        /* === APP HEADER === */
        .wc-header{display:flex;align-items:center;justify-content:space-between;
          padding:14px 0;border-bottom:1px solid var(--wc-bd);}
        .wc-brand{display:flex;align-items:center;gap:10px;
          font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;
          letter-spacing:0.5px;color:var(--wc-txt)}
        .wc-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
        .wc-dot-green{background:#3BD68B}
        .wc-api-status{font-size:13px;color:var(--wc-txt2)}

        /* === COUNTDOWN BANNER === */
        .wc-banner{
          margin:0 -3rem;background:var(--wc-card);border-bottom:1px solid var(--wc-bd);
          position:relative;overflow:hidden;
        }
        .wc-banner::before{
          content:'';position:absolute;inset:0;pointer-events:none;
          background:repeating-linear-gradient(
            180deg,
            rgba(127,184,62,.055) 0,rgba(127,184,62,.055) 28px,
            rgba(127,184,62,.018) 28px,rgba(127,184,62,.018) 56px);
        }
        .wc-banner-inner{
          display:grid;grid-template-columns:auto 1fr auto;
          align-items:center;gap:40px;padding:20px 3rem;position:relative;z-index:1;
        }
        .wc-tourn-lbl{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
          color:var(--wc-txt3);margin-bottom:4px}
        .wc-tourn-name{font-family:'Barlow Condensed',sans-serif;font-size:30px;font-weight:800;
          letter-spacing:1px;line-height:1;color:var(--wc-txt)}
        .wc-tourn-dates{font-size:13px;color:var(--wc-txt3);margin-top:5px}
        .wc-countdown{display:flex;align-items:flex-end;gap:2px;justify-content:center}
        .wc-cd-unit{text-align:center;padding:0 6px}
        .wc-cd-val{display:block;font-family:'Barlow Condensed',sans-serif;font-size:52px;
          font-weight:800;color:var(--wc-gtxt);line-height:1}
        .wc-cd-lbl{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
          color:var(--wc-txt3);display:block;margin-top:3px}
        .wc-cd-sep{font-family:'Barlow Condensed',sans-serif;font-size:40px;font-weight:800;
          color:var(--wc-gline);margin-bottom:14px;line-height:1}

        /* === TICKER === */
        .wc-ticker-outer{border-left:1px solid var(--wc-bd);padding-left:24px;overflow:hidden}
        .wc-ticker-lbl{font-size:10px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
          color:var(--wc-green);margin-bottom:7px}
        .wc-ticker-mask{overflow:hidden}
        .wc-ticker-track{display:flex;gap:40px;white-space:nowrap;
          animation:wc-ticker 42s linear infinite;width:max-content}
        .wc-ticker-item{display:inline-flex;align-items:center;gap:8px;
          font-size:14px;color:var(--wc-txt2);flex-shrink:0}
        .wc-ticker-item strong{color:var(--wc-txt);font-weight:600}
        .wc-tick-val{font-family:'Barlow Condensed',sans-serif;font-size:17px;font-weight:700;color:var(--wc-gtxt)}
        @keyframes wc-ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}

        /* === ST.TABS OVERRIDES === */
        .stTabs [data-baseweb="tab-list"]{
          background:var(--wc-bg)!important;border-bottom:1px solid var(--wc-bd)!important;gap:0;padding:0}
        .stTabs [data-baseweb="tab"]{
          color:var(--wc-txt2)!important;font-family:'Barlow',sans-serif!important;
          font-size:14px!important;font-weight:500!important;
          padding:13px 20px!important;background:transparent!important}
        .stTabs [data-baseweb="tab"]:hover{color:var(--wc-txt)!important}
        .stTabs [aria-selected="true"]{color:var(--wc-gtxt)!important}
        .stTabs [data-baseweb="tab-highlight"]{background-color:var(--wc-green)!important;height:2px!important}
        .stTabs [data-baseweb="tab-border"]{background-color:var(--wc-bd)!important}
        .stTabs [data-baseweb="tab-panel"]{padding:28px 0 0!important}

        /* === KPI GRID === */
        .wc-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}
        .wc-kpi{background:var(--wc-card);border:1px solid var(--wc-bd);border-radius:var(--wc-r);padding:18px 20px}
        .wc-kpi .lbl{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
          color:var(--wc-txt3);margin-bottom:10px}
        .wc-kpi .val{font-family:'Barlow Condensed',sans-serif;font-size:42px;font-weight:800;
          color:var(--wc-txt);line-height:1;display:flex;align-items:center;gap:8px}
        .wc-kpi .sub{font-size:13px;color:var(--wc-txt3);margin-top:6px}

        /* === PITCH DIVIDER === */
        .wc-pitch-div{margin:6px 0 22px;height:24px}
        .wc-pitch-div svg{width:100%;height:100%;display:block}

        /* === SECTION HEADER === */
        .wc-section{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
        .wc-section h3{font-family:'Barlow Condensed',sans-serif;font-size:22px;font-weight:700;
          color:var(--wc-txt);white-space:nowrap;margin:0}
        .wc-legend{display:flex;gap:16px;font-size:13px;color:var(--wc-txt2)}
        .wc-legend span{display:flex;align-items:center;gap:6px}
        .wc-swatch{width:10px;height:10px;border-radius:2px;flex-shrink:0;display:inline-block}

        /* === ODDS ROWS === */
        .wc-row{display:flex;align-items:center;gap:12px;padding:8px 0;
          border-bottom:1px solid rgba(52,56,60,.5)}
        .wc-row:last-child{border-bottom:none}
        .wc-rank{width:22px;text-align:right;font-family:'Barlow Condensed',sans-serif;
          font-size:14px;color:var(--wc-txt3);flex-shrink:0}
        .wc-team{width:190px;flex-shrink:0}
        .wc-team .nm{font-size:15px;color:var(--wc-txt);font-weight:500}
        .wc-team .cf{font-size:12px;color:var(--wc-txt3)}
        .wc-track{flex:1;height:26px;background:var(--wc-track);border-radius:6px;overflow:hidden;
          display:flex;align-items:stretch}
        .wc-fill{flex:0 0 auto;border-radius:6px;
          animation:wc-bar-grow .9s cubic-bezier(.34,1.08,.44,1) forwards}
        .wc-pct{width:60px;text-align:right;font-family:'Barlow Condensed',sans-serif;
          font-size:18px;font-weight:700;color:var(--wc-txt);flex-shrink:0}
        @keyframes wc-bar-grow{
          from{clip-path:inset(0 100% 0 0 round 6px)}
          to  {clip-path:inset(0 0%   0 0 round 6px)}}
        .wc-divider{display:flex;align-items:center;gap:10px;padding:10px 0;
          font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--wc-txt3)}
        .wc-divider .ln{flex:1;border-top:1px dashed var(--wc-bd)}

        /* === MATCH PREDICTOR === */
        .wc-split{display:flex;height:52px;border-radius:10px;overflow:hidden;margin-bottom:8px}
        .wc-split div{display:flex;align-items:center;justify-content:center;
          font-family:'Barlow Condensed',sans-serif;font-size:20px;font-weight:700;min-width:42px}
        .wc-splitlbl{display:flex;justify-content:space-between;font-size:12px;
          color:var(--wc-txt3);margin-bottom:22px}
        .wc-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}
        .wc-card{background:var(--wc-card);border:1px solid var(--wc-bd);
          border-radius:var(--wc-r);padding:20px}
        .wc-card .hd{display:flex;align-items:center;gap:8px;font-size:13px;
          color:var(--wc-txt2);margin-bottom:12px}
        .wc-card .big{font-family:'Barlow Condensed',sans-serif;font-size:50px;font-weight:800;line-height:1}
        .wc-card .mini{height:5px;background:var(--wc-track);border-radius:3px;margin-top:12px;overflow:hidden}
        .wc-context{display:flex;align-items:center;gap:8px;font-size:14px;
          color:var(--wc-txt2);margin-bottom:18px;justify-content:center}
        .wc-context strong{color:var(--wc-txt)}
        .wc-callout{display:flex;align-items:flex-start;gap:12px;background:var(--wc-gtint);
          border-radius:var(--wc-r);padding:14px 18px;font-size:14px;color:#CDE6A8}

        /* === GROUP STANDINGS === */
        .wc-groups{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
        .wc-group{background:var(--wc-card);border:1px solid var(--wc-bd);
          border-radius:var(--wc-r);overflow:hidden}
        .wc-group-hd{padding:10px 14px;border-bottom:1px solid var(--wc-bd);
          background:rgba(127,184,62,.06);border-top:2px solid var(--wc-gline)}
        .wc-group-ltr{font-family:'Barlow Condensed',sans-serif;font-size:15px;
          font-weight:800;letter-spacing:1.5px;color:var(--wc-gtxt)}
        .wc-gteam{display:flex;align-items:center;gap:8px;padding:9px 14px;
          border-bottom:1px solid rgba(52,56,60,.4);font-size:14px}
        .wc-gteam:last-child{border-bottom:none}
        .wc-gteam .nm{flex:1;color:var(--wc-txt2);font-weight:400}
        .wc-gteam .pct{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:700;color:var(--wc-txt3)}
        .wc-gteam.top2 .nm{color:var(--wc-txt);font-weight:500}
        .wc-gteam.top2 .pct{color:var(--wc-goldtxt)}

        /* === ODDS TRACKER === */
        .wc-movers{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;
          max-width:680px;margin-bottom:28px}
        .wc-mover{background:var(--wc-card);border-radius:var(--wc-r);padding:16px 18px}
        .wc-mover .lbl{font-size:12px;color:var(--wc-txt2);margin-bottom:10px;
          display:flex;align-items:center;gap:6px}
        .wc-mover .bd{display:flex;align-items:center;gap:10px;font-size:17px;
          font-weight:500;color:var(--wc-txt)}
        .wc-chips{display:flex;flex-wrap:wrap;gap:9px;margin:4px 0}
        .wc-chip{display:flex;align-items:center;gap:8px;background:var(--wc-track);
          border-radius:16px;padding:6px 11px 6px 8px;font-size:14px;color:var(--wc-txt)}

        /* === SHARED === */
        .wc-eyebrow{font-size:14px;color:var(--wc-txt2);margin:0 0 22px}
        .wc-foot{font-size:12px;color:var(--wc-txt3);padding-top:14px;
          margin-top:16px;border-top:.5px solid var(--wc-bd)}

        /* === RESPONSIVE: TABLET (max-width 1024px) === */
        @media (max-width:1024px){
          .block-container{padding-left:1.5rem;padding-right:1.5rem}
          [data-testid="stCustomComponentV1"]{margin:0 -1.5rem!important;width:calc(100% + 3rem)!important}
          .wc-banner{margin:0 -1.5rem}
          .wc-banner-inner{padding:20px 1.5rem}
          .wc-kpis{grid-template-columns:repeat(2,1fr)}
          .wc-groups{grid-template-columns:repeat(2,1fr)}
        }

        /* === RESPONSIVE: PHONE (max-width 600px) === */
        @media (max-width:600px){
          .block-container{padding-left:1rem;padding-right:1rem}
          [data-testid="stCustomComponentV1"]{margin:0 -1rem!important;width:calc(100% + 2rem)!important}
          .wc-banner{margin:0 -1rem}
          .wc-banner-inner{padding:16px 1rem}

          /* Grids collapse */
          .wc-cards{grid-template-columns:1fr}
          .wc-movers{grid-template-columns:1fr}
          .wc-groups{grid-template-columns:1fr}
          .wc-kpis{grid-template-columns:repeat(2,1fr)}

          /* Shrink display type (!important beats inline font-size in app.py) */
          .wc-kpi .val{font-size:28px!important}
          .wc-card .big{font-size:38px}
          .wc-split{height:44px}
          .wc-split div{font-size:16px}

          /* Title-race rows — tightest layout */
          .wc-team{width:auto;flex:1;min-width:0}
          .wc-team .nm{font-size:13px;white-space:nowrap;overflow:hidden;
            text-overflow:ellipsis;display:block}
          .wc-team .cf{display:none}
          .wc-pct{width:46px;font-size:15px}
          .wc-track{height:22px;min-width:60px}

          /* Header */
          .wc-header{flex-wrap:wrap;gap:6px}
          .wc-api-status{font-size:12px}
          .wc-api-ts{display:none}

          /* Tabs (BaseWeb tab-list scrolls horizontally on overflow) */
          .stTabs [data-baseweb="tab"]{padding:10px 12px!important;font-size:13px!important}

          /* Section header */
          .wc-section{flex-wrap:wrap;gap:8px}
          .wc-section h3{white-space:normal}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
