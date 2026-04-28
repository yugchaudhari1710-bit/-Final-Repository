import math
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

st.title("Drilling Trajectory Visualization — L, J & S Type Profiles")

# -------- INPUT --------
st.sidebar.header("Input Coordinates")

trajectory_type = st.sidebar.selectbox(
    "Trajectory Type",
    ["L-Type (Type I)", "J-Type (Type II)", "S-Type (Type III)"],
)

input_method = st.sidebar.radio(
    "Input Method",
    ["Surface Northing/Easting", "Horizontal Displacement"],
)

Ns = st.sidebar.number_input("Surface Northing (Na)", value=0.0)
Es = st.sidebar.number_input("Surface Easting (Ea)", value=0.0)
Zs = st.sidebar.number_input("Surface TVD", value=0.0)

Vb = st.sidebar.number_input("TVD to KOP (Vb)", value=1000.0)

if input_method == "Surface Northing/Easting":
    Nt = st.sidebar.number_input("Target Northing (Nt)", value=2000.0)
    Et = st.sidebar.number_input("Target Easting (Et)", value=0.0)
    Vt = st.sidebar.number_input("Target TVD (Vt)", value=4820.0)
else:
    H_t = st.sidebar.number_input("Horizontal Displacement (ft)", value=2000.0)
    beta_deg = st.sidebar.number_input("Bearing (°)", value=0.0)
    Vt = st.sidebar.number_input("Target TVD (Vt)", value=4820.0)
    beta_rad = math.radians(beta_deg)
    Nt = Ns + H_t * math.cos(beta_rad)
    Et = Es + H_t * math.sin(beta_rad)

build_rate = st.sidebar.number_input("Build Rate (°/100 ft)", value=1.5)

if trajectory_type == "S-Type (Type III)":
    drop_rate = st.sidebar.number_input("Drop Rate (°/100 ft)", value=3.0)
    max_inc = st.sidebar.number_input("Max Inclination (°)", value=45.0)
else:
    drop_rate = None
    max_inc = None

step = st.sidebar.number_input("Step (ft)", value=50)

surface = (Ns, Es, Zs)
target = (Nt, Et, Vt)


# -------- CORE FUNCTION --------
def generate_well_trajectory(surface, Vb, target, phi, step, t_type, drop_rate=None, max_inc=None):

    Na, Ea, Zs = surface
    Nt, Et, Vt = target

    H_t = math.sqrt((Nt - Na)**2 + (Et - Ea)**2)
    beta = math.atan2(Et - Ea, Nt - Na)

    R = 18000 / (math.pi * phi)

    # ---------- L TYPE ----------
    if t_type == "L-Type (Type I)":
        dV = Vt - Vb
        alpha = math.atan2(H_t, dV)

        MD_kop = Vb
        MD_build = MD_kop + 100 * math.degrees(alpha) / phi
        MD_target = MD_build + dV / math.cos(alpha)

        sections = [
            ("Vertical", 0, MD_kop),
            ("Build", MD_kop, MD_build),
            ("Hold", MD_build, MD_target),
        ]

    # ---------- J TYPE ----------
    elif t_type == "J-Type (Type II)":
        dV = Vt - Zs
        alpha = math.atan2(H_t, dV)

        MD_build = 100 * math.degrees(alpha) / phi
        MD_target = MD_build + dV / math.cos(alpha)

        sections = [
            ("Build", 0, MD_build),
            ("Hold", MD_build, MD_target),
        ]

    # ---------- S TYPE ----------
    else:
        alpha_max = math.radians(max_inc)
        R_drop = 18000 / (math.pi * drop_rate)

        MD_kop = Vb
        MD_build = MD_kop + 100 * max_inc / phi

        V_build = R * math.sin(alpha_max)
        H_build = R * (1 - math.cos(alpha_max))

        if H_t <= H_build:
            raise ValueError("Increase horizontal distance or reduce build rate")

        def residual(a):
            H_drop = R_drop * (math.cos(a) - math.cos(alpha_max))
            V_drop = R_drop * (math.sin(alpha_max) - math.sin(a))
            return (H_t - H_build - H_drop) - (Vt - Zs - V_build - V_drop) * math.tan(a)

        low, high = 0.0001, alpha_max - 0.0001
        for _ in range(80):
            mid = (low + high) / 2
            if residual(low) * residual(mid) <= 0:
                high = mid
            else:
                low = mid

        alpha = mid

        MD_drop = 100 * math.degrees(alpha_max - alpha) / drop_rate
        MD_hold1 = 500
        MD_hold2 = 500

        MD_hold1_end = MD_build + MD_hold1
        MD_drop_end = MD_hold1_end + MD_drop
        MD_target = MD_drop_end + MD_hold2

        sections = [
            ("Vertical", 0, MD_kop),
            ("Build", MD_kop, MD_build),
            ("Hold_max", MD_build, MD_hold1_end),
            ("Drop", MD_hold1_end, MD_drop_end),
            ("Hold_final", MD_drop_end, MD_target),
        ]

    # ---------- WALK ----------
    BR = math.radians(phi) / 100
    DR = math.radians(drop_rate) / 100 if drop_rate else 0

    MD = 0
    inc = 0
    N, E, Z = Na, Ea, Zs
    data = []

    while MD < MD_target:

        for sec, s, e in sections:
            if s <= MD < e:
                if sec == "Vertical":
                    inc_next = 0
                elif sec == "Build":
                    inc_next = inc + BR * step
                elif sec == "Hold_max":
                    inc_next = alpha_max
                elif sec == "Drop":
                    inc_next = inc - DR * step
                else:
                    inc_next = alpha
                section = sec
                break

        next_MD = min(MD + step, MD_target)
        dMD = next_MD - MD

        Z += dMD * math.cos(inc)
        N += dMD * math.sin(inc) * math.cos(beta)
        E += dMD * math.sin(inc) * math.sin(beta)

        MD = next_MD
        inc = inc_next

        data.append({
            "MD": round(MD, 2),
            "Inc": round(math.degrees(inc), 2),
            "N": round(N, 2),
            "E": round(E, 2),
            "TVD": round(Z, 2),
            "Section": section,
        })

    return pd.DataFrame(data)


# -------- RUN --------
try:
    df = generate_well_trajectory(surface, Vb, target, build_rate, step, trajectory_type, drop_rate, max_inc)

    st.subheader("3D Plot")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "Vertical": "blue",
        "Build": "orange",
        "Hold": "green",
        "Hold_max": "purple",
        "Drop": "red",
        "Hold_final": "green",
    }

    for sec in df["Section"].unique():
        d = df[df["Section"] == sec]
        ax.plot(d["N"], d["E"], d["TVD"], color=colors.get(sec, "black"), label=sec)

    ax.invert_zaxis()
    ax.legend()

    st.pyplot(fig)

    st.dataframe(df)

except Exception as e:
    st.error(f"Error: {e}")
