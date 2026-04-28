import math
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

st.title("Drilling Trajectory Visualization — L-Type (Type I Profile)")

# -------- INPUT --------
st.sidebar.header("Input Coordinates")

trajectory_type = st.sidebar.selectbox(
    "Trajectory Type",
    ["L-Type (Type I)", "J-Type (Type II)", "S-Type (Type III)"],
    index=0
)

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Surface Northing/Easting", "Horizontal Displacement"],
    index=0
)

Ns = st.sidebar.number_input("Surface Northing (Na)", value=0.0)
Es = st.sidebar.number_input("Surface Easting (Ea)", value=0.0)
Zs = st.sidebar.number_input("Surface TVD", value=0.0)

Vb = st.sidebar.number_input("TVD to KOP (Vb)", value=1000.0)

if input_method == "Surface Northing/Easting":
    Nt = st.sidebar.number_input("Target Northing (Nt)", value=1200.0)
    Et = st.sidebar.number_input("Target Easting (Et)", value=800.0)
    Vt = st.sidebar.number_input("Target TVD (Vt)", value=4820.0)

    # Calculate horizontal displacement and bearing from coordinates
    H_t = math.sqrt((Nt - Ns)**2 + (Et - Es)**2)
    beta_deg = math.degrees(math.atan2(Et - Es, Nt - Ns))

else:  # Horizontal Displacement
    H_t = st.sidebar.number_input("Horizontal Displacement (H_t, ft)", value=1200.0, min_value=0.0)
    beta_deg = st.sidebar.number_input("Target Bearing (β, °)", value=33.7, min_value=-180.0, max_value=180.0)
    Vt = st.sidebar.number_input("Target TVD (Vt)", value=4820.0)

    # Calculate target coordinates from horizontal displacement and bearing
    beta_rad = math.radians(beta_deg)
    Nt = Ns + H_t * math.cos(beta_rad)
    Et = Es + H_t * math.sin(beta_rad)

build_rate = st.sidebar.number_input("Build Rate φ (°/100 ft)", value=1.5, min_value=0.01)

if trajectory_type == "S-Type (Type III)":
    drop_rate = st.sidebar.number_input("Drop Rate (°/100 ft)", value=1.0, min_value=0.01)
    max_inclination = st.sidebar.number_input("Max Inclination (°)", value=45.0, min_value=0.01, max_value=90.0)
else:
    drop_rate = None
    max_inclination = None

step = st.sidebar.number_input("Survey Step (ft)", value=50, min_value=1)

surface = (Ns, Es, Zs)
target = (Nt, Et, Vt)

# -------- TRAJECTORY FUNCTION --------
def generate_well_trajectory(surface, Vb, target, phi, step=50, trajectory_type="L-Type (Type I)", drop_rate=None, max_inclination=None):
    """
    Generate well trajectory for different profile types.

    Type I (L-Type): Vertical + Build + Hold
    Type II (J-Type): Build + Hold
    Type III (S-Type): Build + Drop + Hold
    """
    Na, Ea, Zs = surface
    Nt, Et, Vt = target

    # ── Step 1: Horizontal displacement H_t ──────────────────────────────────
    # Formula: H_t = sqrt((N_t - N_a)^2 + (E_t - E_a)^2)
    H_t = math.sqrt((Nt - Na)**2 + (Et - Ea)**2)

    # ── Step 2: Target bearing β ──────────────────────────────────────────────
    # Formula: β = tan⁻¹((E_t - E_a) / (N_t - N_a))
    beta_rad = math.atan2(Et - Ea, Nt - Na)
    beta_deg = math.degrees(beta_rad)

    # ── Step 3: Radius of curvature R ────────────────────────────────────────
    # Formula: R = 18000 / (π * φ)
    R = 18000.0 / (math.pi * phi)

    if trajectory_type == "L-Type (Type I)":
        # Original L-Type calculations
        dV = Vt - Vb
        dH_minus_R = H_t - R

        PT = math.sqrt(dH_minus_R**2 + dV**2)

        x_rad = math.atan2(dH_minus_R, dV)
        y_rad = math.asin((R * math.cos(x_rad)) / PT)

        alpha_rad = x_rad + y_rad
        alpha_deg = math.degrees(alpha_rad)

        Vc = Vb + R * math.sin(alpha_rad)
        Hc = R * (1 - math.cos(alpha_rad))

        MD_b = Vb
        BC = 100.0 * alpha_deg / phi
        MD_c = MD_b + BC
        MD_t = MD_c + (Vt - Vc) / math.cos(alpha_rad)

        # Trajectory sections
        sections = [
            ("Vertical", 0, MD_b),
            ("Build", MD_b, MD_c),
            ("Hold", MD_c, MD_t)
        ]

    elif trajectory_type == "J-Type (Type II)":
        # J-Type: Build from surface to target inclination, then hold
        # Calculate required inclination to reach target
        dV = Vt - Zs  # Total vertical distance from surface
        alpha_rad = math.atan2(H_t, dV)
        alpha_deg = math.degrees(alpha_rad)

        # Build section to reach alpha
        BC = 100.0 * alpha_deg / phi
        MD_c = BC
        MD_t = MD_c + (Vt - Zs - R * math.sin(alpha_rad)) / math.cos(alpha_rad)

        Vc = Zs + R * math.sin(alpha_rad)
        Hc = R * (1 - math.cos(alpha_rad))

        MD_b = 0  # No vertical section

        sections = [
            ("Build", 0, MD_c),
            ("Hold", MD_c, MD_t)
        ]

    elif trajectory_type == "S-Type (Type III)":
        # S-Type: Build to max inclination, drop to final inclination, then hold
        if max_inclination is None or drop_rate is None:
            raise ValueError("S-Type requires max_inclination and drop_rate parameters")

        alpha_max_rad = math.radians(max_inclination)
        R_drop = 18000.0 / (math.pi * drop_rate)

        # Build to maximum inclination
        MD_build = 100.0 * max_inclination / phi
        V_build = R * math.sin(alpha_max_rad)
        H_build = R * (1 - math.cos(alpha_max_rad))

        dV_total = Vt - Zs

        def s_type_residual(alpha_final):
            # Drop section horizontal and vertical changes from alpha_max to alpha_final
            H_drop = R_drop * (math.cos(alpha_final) - math.cos(alpha_max_rad))
            V_drop = R_drop * (math.sin(alpha_max_rad) - math.sin(alpha_final))
            H_remaining = H_t - H_build - H_drop
            V_remaining = dV_total - V_build - V_drop
            return H_remaining - V_remaining * math.tan(alpha_final)

        # Solve final inclination for S-Type using a simple bisection method
        alpha_low = 1e-6
        alpha_high = alpha_max_rad - 1e-6
        residual_low = s_type_residual(alpha_low)
        residual_high = s_type_residual(alpha_high)

        if residual_low * residual_high > 0:
            raise ValueError("S-Type geometry cannot satisfy target with the given max inclination and drop rate.")

        for _ in range(80):
            alpha_mid = 0.5 * (alpha_low + alpha_high)
            residual_mid = s_type_residual(alpha_mid)
            if abs(residual_mid) < 1e-6:
                break
            if residual_low * residual_mid <= 0:
                alpha_high = alpha_mid
                residual_high = residual_mid
            else:
                alpha_low = alpha_mid
                residual_low = residual_mid

        alpha_final_rad = alpha_mid
        alpha_final_deg = math.degrees(alpha_final_rad)

        H_drop = R_drop * (math.cos(alpha_final_rad) - math.cos(alpha_max_rad))
        V_drop = R_drop * (math.sin(alpha_max_rad) - math.sin(alpha_final_rad))

        MD_drop = 100.0 * math.degrees(alpha_max_rad - alpha_final_rad) / drop_rate
        MD_c = MD_build
        MD_d = MD_c + MD_drop

        H_remaining = H_t - H_build - H_drop
        V_remaining = dV_total - V_build - V_drop

        MD_t = MD_d + V_remaining / math.cos(alpha_final_rad)

        # Define variables for consistency
        alpha_rad = alpha_final_rad
        alpha_deg = alpha_final_deg
        MD_b = 0  # No vertical section

        # Define Vc and Hc for S-Type (at end of build section)
        Vc = V_build
        Hc = H_build
        BC = MD_build  # Build curve length

        sections = [
            ("Build", 0, MD_build),
            ("Drop", MD_build, MD_d),
            ("Hold", MD_d, MD_t)
        ]

    # ── Step 7: Walk the wellbore step-by-step ───────────────────────────────
    BR_rad_per_ft = math.radians(phi) / 100.0   # build rate in rad/ft
    if drop_rate:
        DR_rad_per_ft = math.radians(drop_rate) / 100.0  # drop rate in rad/ft
    else:
        DR_rad_per_ft = None

    MD   = 0.0
    N, E, Z = Na, Ea, Zs
    inc_rad = 0.0
    data = []

    while MD < MD_t - 1e-6:
        # Determine section and inclination at current MD
        section = None
        inc_next = 0.0

        for sec_name, sec_start, sec_end in sections:
            if sec_start <= MD < sec_end:
                section = sec_name
                if sec_name == "Vertical":
                    inc_next = 0.0
                elif sec_name == "Build":
                    if trajectory_type == "J-Type (Type II)":
                        s = MD  # distance from start
                    elif trajectory_type == "S-Type (Type III)":
                        s = MD  # distance from start
                    else:
                        s = MD - MD_b
                    inc_next = BR_rad_per_ft * s
                    if trajectory_type == "S-Type (Type III)":
                        inc_next = min(inc_next, alpha_max_rad)
                    elif trajectory_type == "L-Type (Type I)" or trajectory_type == "J-Type (Type II)":
                        inc_next = min(inc_next, alpha_rad)
                elif sec_name == "Drop":
                    s = MD - MD_c
                    inc_next = alpha_max_rad - DR_rad_per_ft * s
                    inc_next = max(inc_next, alpha_final_rad)
                elif sec_name == "Hold":
                    if trajectory_type == "L-Type (Type I)" or trajectory_type == "J-Type (Type II)":
                        inc_next = alpha_rad
                    elif trajectory_type == "S-Type (Type III)":
                        inc_next = alpha_final_rad
                break

        # Clamp inclination
        if trajectory_type == "L-Type (Type I)":
            inc_next = min(inc_next, alpha_rad)
        elif trajectory_type == "J-Type (Type II)":
            inc_next = min(inc_next, alpha_rad)
        elif trajectory_type == "S-Type (Type III)":
            if section == "Build":
                inc_next = min(inc_next, alpha_max_rad)
            elif section == "Drop":
                inc_next = max(inc_next, alpha_final_rad)
            elif section == "Hold":
                inc_next = alpha_final_rad

        # Next station
        next_MD    = min(MD + step, MD_t)
        delta_MD   = next_MD - MD

        # ── Minimum curvature position update ────────────────────────────────
        theta1, theta2 = inc_rad, inc_next
        phi1 = phi2 = beta_rad  # azimuth fixed toward target

        delta_theta  = theta2 - theta1
        sin_t1, sin_t2 = math.sin(theta1), math.sin(theta2)
        cos_t1, cos_t2 = math.cos(theta1), math.cos(theta2)

        # Dogleg angle for minimum curvature
        beta_angle = math.acos(
            max(-1.0, min(1.0,
                math.cos(theta1) * math.cos(theta2) + sin_t1 * sin_t2 * math.cos(phi2 - phi1)
            ))
        )

        RF = (2.0 / beta_angle) * math.tan(beta_angle / 2) if beta_angle != 0 else 1.0

        delta_TVD = (delta_MD / 2) * (cos_t1 + cos_t2) * RF
        delta_N   = (delta_MD / 2) * (sin_t1 * math.cos(phi1) + sin_t2 * math.cos(phi2)) * RF
        delta_E   = (delta_MD / 2) * (sin_t1 * math.sin(phi1) + sin_t2 * math.sin(phi2)) * RF

        Z   += delta_TVD
        N   += delta_N
        E   += delta_E
        MD   = next_MD
        inc_rad = inc_next

        data.append({
            "MD (ft)"          : round(MD, 2),
            "Inclination (°)"  : round(math.degrees(inc_rad), 4),
            "Azimuth (°)"      : round(beta_deg, 4),
            "Northing (ft)"    : round(N, 4),
            "Easting (ft)"     : round(E, 4),
            "TVD (ft)"         : round(Z, 4),
            "Section"          : section,
        })

    df = pd.DataFrame(data)

    # ── Summary results ───────────────────────────────────────────────────────
    if trajectory_type == "L-Type (Type I)":
        summary = {
            "H_t (Horizontal Displacement, ft)": round(H_t, 2),
            "β (Target Bearing, °)"            : round(beta_deg, 4),
            "R (Radius of Curvature, ft)"      : round(R, 2),
            "α (Max Inclination, °)"           : round(alpha_deg, 4),
            "V_c (TVD at Point C, ft)"         : round(Vc, 2),
            "H_c (Horizontal Departure at C, ft)": round(Hc, 2),
            "MD at KOP (ft)"                   : round(MD_b, 2),
            "Arc BC (ft)"                      : round(BC, 2),
            "MD at C (ft)"                     : round(MD_c, 2),
            "MD at Target (ft)"                : round(MD_t, 2),
        }
    elif trajectory_type == "J-Type (Type II)":
        summary = {
            "H_t (Horizontal Displacement, ft)": round(H_t, 2),
            "β (Target Bearing, °)"            : round(beta_deg, 4),
            "R (Radius of Curvature, ft)"      : round(R, 2),
            "α (Max Inclination, °)"           : round(alpha_deg, 4),
            "V_c (TVD at Point C, ft)"         : round(Vc, 2),
            "H_c (Horizontal Departure at C, ft)": round(Hc, 2),
            "MD at KOP (ft)"                   : round(MD_b, 2),
            "Arc BC (ft)"                      : round(BC, 2),
            "MD at C (ft)"                     : round(MD_c, 2),
            "MD at Target (ft)"                : round(MD_t, 2),
        }
    elif trajectory_type == "S-Type (Type III)":
        summary = {
            "H_t (Horizontal Displacement, ft)": round(H_t, 2),
            "β (Target Bearing, °)"            : round(beta_deg, 4),
            "R (Build Radius, ft)"             : round(R, 2),
            "R (Drop Radius, ft)"              : round(R_drop, 2),
            "α_max (Max Inclination, °)"       : round(max_inclination, 4),
            "α_final (Final Inclination, °)"   : round(alpha_deg, 4),
            "V_c (TVD at End of Build, ft)"    : round(Vc, 2),
            "H_c (Horizontal at End of Build, ft)": round(Hc, 2),
            "MD at KOP (ft)"                   : round(MD_b, 2),
            "Build Section (ft)"               : round(BC, 2),
            "Drop Section (ft)"                : round(MD_drop, 2),
            "MD at Target (ft)"                : round(MD_t, 2),
        }

    return df, summary


# -------- RUN --------
try:
    df, summary = generate_well_trajectory(
        surface, Vb, target, build_rate, int(step),
        trajectory_type, drop_rate, max_inclination
    )

    # -------- INPUT SUMMARY --------
    st.subheader("📍 Input Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Input Method", input_method)
        st.metric("Trajectory Type", trajectory_type)
        st.metric("Surface Northing (Na)", f"{Ns:.1f} ft")
        st.metric("Surface Easting (Ea)", f"{Es:.1f} ft")
        st.metric("Surface TVD", f"{Zs:.1f} ft")
        st.metric("TVD to KOP (Vb)", f"{Vb:.1f} ft")

    with col2:
        if input_method == "Surface Northing/Easting":
            st.metric("Target Northing (Nt)", f"{Nt:.1f} ft")
            st.metric("Target Easting (Et)", f"{Et:.1f} ft")
        else:
            st.metric("Horizontal Displacement (H_t)", f"{H_t:.1f} ft")
            st.metric("Target Bearing (β)", f"{beta_deg:.1f}°")
        st.metric("Target TVD (Vt)", f"{Vt:.1f} ft")
        st.metric("Build Rate φ", f"{build_rate:.2f}°/100 ft")
        if trajectory_type == "S-Type (Type III)":
            st.metric("Drop Rate", f"{drop_rate:.2f}°/100 ft")
            st.metric("Max Inclination", f"{max_inclination:.1f}°")

    # -------- SUMMARY --------
    st.subheader("📐 Calculated Parameters")
    col1, col2 = st.columns(2)
    items = list(summary.items())
    for i, (k, v) in enumerate(items):
        (col1 if i < len(items)//2 else col2).metric(k, v)

    # -------- PLOT --------
    st.subheader("3D Well Trajectory")
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Color mapping for different trajectory types
    if trajectory_type == "L-Type (Type I)":
        color_map = {"Vertical": "blue", "Build": "orange", "Hold": "green"}
    elif trajectory_type == "J-Type (Type II)":
        color_map = {"Build": "orange", "Hold": "green"}
    elif trajectory_type == "S-Type (Type III)":
        color_map = {"Build": "orange", "Drop": "red", "Hold": "green"}

    for sec in df["Section"].unique():
        subset = df[df["Section"] == sec]
        ax.plot(subset["Northing (ft)"], subset["Easting (ft)"], subset["TVD (ft)"],
                label=sec, color=color_map.get(sec, "gray"), linewidth=2)

    ax.set_xlabel("Northing (ft)")
    ax.set_ylabel("Easting (ft)")
    ax.set_zlabel("TVD (ft)")
    ax.set_title(f"{trajectory_type} Well Trajectory")
    ax.legend()
    ax.invert_zaxis()

    st.pyplot(fig)

    # -------- TABLE --------
    st.subheader("📋 Survey Table")
    st.dataframe(df, use_container_width=True)

    st.success("App Running Successfully ✅")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Tip: Check that H_t > R (target must be beyond the build arc radius). "
            "Try increasing the target horizontal distance or reducing the build rate.")
