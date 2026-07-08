import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Well Performance Simulator", layout="wide")
st.title("Multiphase Well Performance Simulator")
st.caption("by ProbablyPushkar")
st.markdown("A 1D steady-state nodal analysis and flow assurance tool.")

# ==========================================
# MODULE B: PVT & FLUID CORRELATIONS
# ==========================================
def calc_rs(p, t_f, api, sg_g):
    if p <= 14.7: return 0.0
    x = 0.0125 * api - 0.00091 * t_f
    return sg_g * (((p / 18.2) + 1.4) * 10**x)**1.2048

def calc_bo(rs, sg_g, api, t_f):
    sg_o = 141.5 / (131.5 + api)
    f = rs * math.sqrt(sg_g / sg_o) + 1.25 * t_f
    return 0.9759 + 0.00012 * f**1.2

def calc_bg(p, t_f):
    t_r = t_f + 460
    z = 0.9 
    return 0.02827 * z * t_r / p

def get_fluid_properties(p, t_f, api, sg_g, wc, gor):
    sg_o = 141.5 / (131.5 + api)
    sg_w = 1.05 
    
    rs_local = calc_rs(p, t_f, api, sg_g)
    rs_actual = min(rs_local, gor) 
    
    bo = calc_bo(rs_actual, sg_g, api, t_f)
    bg = calc_bg(p, t_f)
    bw = 1.02 
    
    free_gas = max(0, gor - rs_actual)
    
    rho_o = (62.4 * sg_o + 0.0136 * rs_actual * sg_g) / bo
    rho_w = 62.4 * sg_w / bw
    rho_g = 0.0764 * sg_g / (bg / 5.615) if bg > 0 else 0 
    
    q_o_std = 1.0 * (1 - wc/100)
    q_w_std = 1.0 * (wc/100)
    
    q_o_local = q_o_std * bo
    q_w_local = q_w_std * bw
    q_g_local = q_o_std * free_gas * bg / 5.615 
    
    q_total = q_o_local + q_w_local + q_g_local
    if q_total == 0:
        return 0, 0, 0, 0
        
    lambda_l = (q_o_local + q_w_local) / q_total
    rho_mix = (q_o_local*rho_o + q_w_local*rho_w + q_g_local*rho_g) / q_total
    
    return rho_mix, lambda_l, q_total, free_gas

# ==========================================
# MODULE C: HYDRAULICS & IPR ENGINES
# ==========================================
def calculate_ipr(p_res, p_b, pi):
    pwf_arr = np.linspace(0, p_res, 50)
    q_arr = []
    
    q_b = pi * (p_res - p_b) if p_res > p_b else 0
    
    for pwf in pwf_arr:
        if pwf >= p_b:
            q = pi * (p_res - pwf)
        else:
            if p_res > p_b:
                q = q_b + (pi * p_b / 1.8) * (1 - 0.2*(pwf/p_b) - 0.8*(pwf/p_b)**2)
            else:
                q_max = pi * p_res / 1.8
                q = q_max * (1 - 0.2*(pwf/p_res) - 0.8*(pwf/p_res)**2)
        q_arr.append(max(0, q))
        
    return np.array(q_arr), pwf_arr

def calculate_pressure_gradient(p, t_f, d_in, rate, api, sg_g, wc, gor):
    area = (math.pi / 4) * (d_in / 12)**2 
    rho_mix, lambda_l, q_total_ratio, free_gas = get_fluid_properties(p, t_f, api, sg_g, wc, gor)
    
    q_total_ft3_sec = rate * q_total_ratio * 5.615 / 86400
    velocity = q_total_ft3_sec / area if area > 0 else 0
    
    dp_dz_elv = rho_mix / 144
    f = 0.02
    dp_dz_fric = (f * rho_mix * velocity**2) / (2 * 32.2 * (d_in / 12) * 144) if d_in > 0 else 0
    
    return dp_dz_elv + dp_dz_fric, velocity, rho_mix

def get_id_at_depth(depth, casings, tbg_depth, tbg_id):
    """Determines the internal diameter for flow at a specific depth"""
    if depth <= tbg_depth:
        return tbg_id
    else:
        valid_casings = [c for c in casings if c["depth"] >= depth]
        if valid_casings:
            return min(valid_casings, key=lambda x: x["id"])["id"]
        # Fallback to open hole or largest casing if below all casing shoes
        return casings[-1]["id"] if casings else tbg_id + 2

def calculate_vlp_curve(p_wh, t_wh, t_bh, md_total, tbg_id, tbg_depth, casings, api, sg_g, wc, gor, rates, n_steps=50):
    p_wf_arr = []
    dz = md_total / n_steps
    
    for q in rates:
        if q == 0:
            p_wf_arr.append(p_wh) 
            continue
            
        p_curr = p_wh
        for step in range(n_steps):
            current_depth = step * dz
            local_id = get_id_at_depth(current_depth, casings, tbg_depth, tbg_id)
            t_curr = t_wh + (t_bh - t_wh) * (current_depth / md_total)
            
            dp_dz, _, _ = calculate_pressure_gradient(p_curr, t_curr, local_id, q, api, sg_g, wc, gor)
            p_curr += dp_dz * dz
        p_wf_arr.append(p_curr)
        
    return np.array(p_wf_arr)

# ==========================================
# MODULE A & SIDEBAR: INPUTS
# ==========================================
st.sidebar.header("Input Parameters")

with st.sidebar.expander("1. Wellbore Geometry", expanded=True):
    md_total = st.number_input("Total Measured Depth (ft)", value=10000.0, step=500.0)
    
    st.markdown("**Casing Design**")
    num_casings = st.number_input("Number of Casings", min_value=0, max_value=4, value=2)
    casings = []
    for i in range(int(num_casings)):
        col1, col2 = st.columns(2)
        # Default wider and shallower for top casings
        c_id = col1.number_input(f"Casing {i+1} ID (in)", value=float(12.0 - i*3.5), step=0.1, key=f"c_id_{i}")
        c_depth = col2.number_input(f"Csg {i+1} Depth (ft)", value=float(2000.0 + i*6000.0), step=500.0, key=f"c_dp_{i}")
        casings.append({"id": c_id, "depth": c_depth})
        
    st.markdown("**Tubing & Packer**")
    tbg_id = st.number_input("Tubing Inner Diameter (in)", value=2.992, step=0.1)
    tbg_od = st.number_input("Tubing Outer Diameter (in)", value=3.5, step=0.1)
    tbg_depth = st.number_input("Tubing Bottom Depth (ft)", value=9000.0, step=500.0)
    has_packer = st.checkbox("Set Packer at Tubing End", value=True)
    
    st.markdown("**Perforations**")
    perf_top = st.number_input("Perforation Top Depth (ft)", value=9500.0, step=100.0)
    perf_bot = st.number_input("Perforation Bottom Depth (ft)", value=9600.0, step=100.0)

with st.sidebar.expander("2. Reservoir & Inflow", expanded=True):
    p_res = st.number_input("Reservoir Pressure (psi)", value=4000.0, step=100.0)
    t_res = st.number_input("Reservoir Temperature (°F)", value=200.0, step=10.0)
    pi = st.number_input("Productivity Index (STB/d/psi)", value=2.5, step=0.1)
    p_b = st.number_input("Bubble Point Pressure (psi)", value=2500.0, step=100.0)

with st.sidebar.expander("3. Fluids & Surface", expanded=True):
    p_wh = st.number_input("Wellhead Pressure (psi)", value=500.0, step=50.0)
    t_wh = st.number_input("Wellhead Temperature (°F)", value=100.0, step=10.0)
    api = st.number_input("Oil API Gravity", value=35.0, step=1.0)
    sg_g = st.number_input("Gas Specific Gravity", value=0.7, step=0.05)
    wc = st.number_input("Water Cut (%)", value=20.0, step=5.0)
    gor = st.number_input("Gas-Oil Ratio (scf/STB)", value=800.0, step=100.0)

# ==========================================
# CALCULATION EXECUTION
# ==========================================
q_ipr, p_ipr = calculate_ipr(p_res, p_b, pi)
max_rate = max(q_ipr)

q_vlp = np.linspace(100, max_rate * 1.2, 20)
p_vlp = calculate_vlp_curve(p_wh, t_wh, t_res, md_total, tbg_id, tbg_depth, casings, api, sg_g, wc, gor, q_vlp)

def find_intersection(q_ipr, p_ipr, q_vlp, p_vlp):
    try:
        f_ipr = interp1d(q_ipr, p_ipr, kind='cubic', fill_value="extrapolate")
        f_vlp = interp1d(q_vlp, p_vlp, kind='cubic', fill_value="extrapolate")
        
        def diff(q): return f_ipr(q) - f_vlp(q)
        q_opt = fsolve(diff, max_rate/2)[0]
        p_opt = f_ipr(q_opt)
        
        if 0 <= q_opt <= max_rate:
            return float(q_opt), float(p_opt)
    except: pass
    return None, None

q_op, p_op = find_intersection(q_ipr, p_ipr, q_vlp, p_vlp)

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["Nodal Analysis", "Wellbore Schematic", "P/T & Flow Assurance", "Sensitivity Analysis"])

# --- TAB 1: NODAL ANALYSIS ---
with tab1:
    st.subheader("System Nodal Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_ipr, y=p_ipr, mode='lines', name='IPR Curve', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=q_vlp, y=p_vlp, mode='lines', name='VLP Curve', line=dict(color='red', width=3)))
    
    if q_op and p_op:
        fig.add_trace(go.Scatter(x=[q_op], y=[p_op], mode='markers', name='Operating Point',
                                 marker=dict(color='green', size=12, symbol='star')))
        st.success(f"**Operating Point:** Flow Rate = {q_op:.1f} STB/d  |  Bottomhole Pressure = {p_op:.1f} psi")
    else:
        st.error("No intersection found. The well cannot flow naturally under these conditions.")

    fig.update_layout(xaxis_title="Flow Rate (STB/d)", yaxis_title="Bottomhole Flowing Pressure (psi)",
                      hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: WELLBORE SCHEMATIC ---
with tab2:
    st.subheader("Dynamic Wellbore Schematic")
    fig_sch = go.Figure()
    
    # Sort casings by ID descending to draw outer (widest) first
    casings_sorted = sorted(casings, key=lambda x: x["id"], reverse=True)
    
    # Draw Open Hole (Background below casing)
    max_casing_id = casings_sorted[0]["id"] if casings_sorted else tbg_od + 4
    fig_sch.add_shape(type="rect", x0=-max_casing_id/2, y0=0, x1=max_casing_id/2, y1=md_total,
                      line=dict(color="brown", width=1), fillcolor="#F4A460", layer="below")
    
    # Draw Casings
    for c in casings_sorted:
        fig_sch.add_shape(type="rect", x0=-c["id"]/2, y0=0, x1=c["id"]/2, y1=c["depth"],
                          line=dict(color="black", width=2), fillcolor="lightgrey")
        
    # Draw Tubing
    fig_sch.add_shape(type="rect", x0=-tbg_od/2, y0=0, x1=tbg_od/2, y1=tbg_depth,
                      line=dict(color="darkblue", width=2), fillcolor="lightblue")
    fig_sch.add_shape(type="rect", x0=-tbg_id/2, y0=0, x1=tbg_id/2, y1=tbg_depth,
                      line=dict(color="darkblue", width=1), fillcolor="white")
                      
    # Draw Packer
    if has_packer and tbg_depth > 0:
        valid_casings = [c for c in casings if c["depth"] >= tbg_depth]
        pack_id = min(valid_casings, key=lambda x: x["id"])["id"] if valid_casings else max_casing_id
        pack_thick = md_total * 0.015 
        
        # Left packer element
        fig_sch.add_shape(type="rect", x0=-pack_id/2, y0=tbg_depth-pack_thick, x1=-tbg_od/2, y1=tbg_depth,
                          line=dict(color="black", width=1), fillcolor="#333333")
        # Right packer element
        fig_sch.add_shape(type="rect", x0=tbg_od/2, y0=tbg_depth-pack_thick, x1=pack_id/2, y1=tbg_depth,
                          line=dict(color="black", width=1), fillcolor="#333333")
                          
    # Draw Perforations
    fig_sch.add_shape(type="rect", x0=-max_casing_id/2 - 1, y0=perf_top, x1=max_casing_id/2 + 1, y1=perf_bot,
                      line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.5)")
    
    # STRICTLY FIX AXIS DIRECTION
    fig_sch.update_layout(xaxis_title="Diameter (in)", yaxis_title="Depth (ft)",
                          yaxis=dict(range=[md_total, 0]), showlegend=False,
                          template="plotly_white", width=400, height=700)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fig_sch, use_container_width=True)
    with col2:
        st.markdown("""
        ### Schematic Details
        * **Brown Zone:** Earth / Open Hole
        * **Outer Light Grey Layers:** Casings (up to 4, dynamically stacked)
        * **Inner Light Blue:** Tubing OD/ID Annulus
        * **Dark Grey Blocks:** Packer (dynamically seals against nearest outer casing)
        * **Red Zone:** Perforated Interval
        
        *The wellbore correctly scales downward, with 0 ft at the top.*
        """)

# --- TAB 3: PROFILE & FLOW ASSURANCE ---
with tab3:
    st.subheader("Pressure/Temperature Gradients & Flow Assurance")
    if q_op:
        depths = np.linspace(0, md_total, 50)
        p_prof, t_prof, v_prof, v_erosional, t_hyd = [], [], [], [], []
        
        p_curr = p_wh
        dz = md_total / 50
        
        for step, d in enumerate(depths):
            t_curr = t_wh + (t_res - t_wh) * (d / md_total)
            local_id = get_id_at_depth(d, casings, tbg_depth, tbg_id)
            
            dp_dz, v_mix, rho_mix = calculate_pressure_gradient(p_curr, t_curr, local_id, q_op, api, sg_g, wc, gor)
            ve = 100 / math.sqrt(rho_mix) if rho_mix > 0 else 999
            th = max(32, 13.47 * math.log(p_curr) + 34.27) if p_curr > 1 else 32
            
            p_prof.append(p_curr)
            t_prof.append(t_curr)
            v_prof.append(v_mix)
            v_erosional.append(ve)
            t_hyd.append(th)
            p_curr += dp_dz * dz
            
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(x=p_prof, y=depths, mode='lines', name='Pressure (psi)', line=dict(color='blue')))
        fig_pt.update_layout(yaxis=dict(range=[md_total, 0]), xaxis_title="Pressure (psi)", yaxis_title="Depth (ft)")
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_prof, y=depths, mode='lines', name='Fluid Temp', line=dict(color='orange')))
        fig_t.add_trace(go.Scatter(x=t_hyd, y=depths, mode='lines', name='Hydrate Formation Temp', line=dict(color='cyan', dash='dash')))
        fig_t.update_layout(yaxis=dict(range=[md_total, 0]), xaxis_title="Temperature (°F)", yaxis_title="Depth (ft)")
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_pt, use_container_width=True)
        col2.plotly_chart(fig_t, use_container_width=True)
        
        st.markdown("### Risk Analysis Flags")
        if any(t <= th for t, th in zip(t_prof, t_hyd)):
            st.error("⚠️ **Hydrate Risk Detected:** Fluid temperature drops below the hydrate formation curve at shallower depths.")
        else:
            st.success("✅ No hydrate risk detected within the wellbore.")
            
        if any(v >= ve for v, ve in zip(v_prof, v_erosional)):
            st.error("⚠️ **Erosional Velocity Exceeded:** Fluid velocity exceeds API RP 14E safe limits.")
        else:
            st.success("✅ Flow velocities are within safe erosional limits.")
    else:
        st.warning("Establish an operating point in Nodal Analysis to view profiles.")

# --- TAB 4: SENSITIVITY ANALYSIS ---
with tab4:
    st.subheader("Batch Sensitivity Analysis")
    sens_param = st.selectbox("Select Parameter to Vary", ["Tubing ID (in)", "Water Cut (%)", "Wellhead Pressure (psi)"])
    
    if sens_param == "Tubing ID (in)":
        sens_vals = [2.441, 2.992, 3.958]
    elif sens_param == "Water Cut (%)":
        sens_vals = [0, 50, 90]
    else:
        sens_vals = [250, 500, 1000]
        
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=q_ipr, y=p_ipr, mode='lines', name='Base IPR', line=dict(color='blue', width=3)))
    
    for val in sens_vals:
        if sens_param == "Tubing ID (in)":
            p_vlp_sens = calculate_vlp_curve(p_wh, t_wh, t_res, md_total, val, tbg_depth, casings, api, sg_g, wc, gor, q_vlp)
        elif sens_param == "Water Cut (%)":
            p_vlp_sens = calculate_vlp_curve(p_wh, t_wh, t_res, md_total, tbg_id, tbg_depth, casings, api, sg_g, val, gor, q_vlp)
        else:
            p_vlp_sens = calculate_vlp_curve(val, t_wh, t_res, md_total, tbg_id, tbg_depth, casings, api, sg_g, wc, gor, q_vlp)
            
        fig_sens.add_trace(go.Scatter(x=q_vlp, y=p_vlp_sens, mode='lines', name=f'VLP: {val}', line=dict(dash='dash')))
        
    fig_sens.update_layout(xaxis_title="Flow Rate (STB/d)", yaxis_title="Bottomhole Flowing Pressure (psi)",
                           hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_sens, use_container_width=True)
