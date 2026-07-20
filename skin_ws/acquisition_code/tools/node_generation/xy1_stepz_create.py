"""새 센서 취득용 node 생성 — xy 1mm 격자 × 각 지점 계단식 z (전진식).

SOP(history/fig_data/ACQUISITION_SOP.md) 확정 조건:
  - xy 1mm 격자 (±10mm, 21×21 = 441점)
  - 각 격자점에서 z 를 0.5mm 단위로 계단 하강하며 각 단계 정지·측정 (전진식)
  - 최대 깊이 2.5mm 마지노선 (⚠️ 네 부 센서 눌림 방지 — 초과 금지)
  - d5 / d10 인덴터 각각 실행(접촉 시작 z 다르면 CONTACT_Z 조정)

기존 SATS_create(단일 깊이)와 달리 각 xy 에서 여러 깊이를 계단식으로 측정한다.
node 행 = [순서, "4직선", X, Y, Z, U, 시작, 가속, 감속, 구동, IO].
  - Z 증가 = 하강(눌림), U 값 = 그 지점에서 대기(측정) 시간.
  - 속도 (1,10,10,10)=쾌속 이동, (0.6667...)=하강.

⚠️ 아래 SAFE_Z / CONTACT_Z 는 **물리 셋업 절대좌표** — 실제 센서 표면 위치에
맞게 반드시 확인·조정할 것. CONTACT_Z + MAX_DEPTH 가 센서 파손 한계를 넘지 않게.
"""
import os

# ── 사용자 셋업에 맞게 확인·조정 (z 절대좌표) ──────────────────────────────
SAVE_DIR   = r"C:\Users\SM\Desktop\node_create"   # 로컬 저장 경로
SAFE_Z     = 13.0    # 안전(비접촉) 높이 — 이동 시
CONTACT_Z  = 13.0    # 접촉 시작 z (센서 표면 닿는 지점, 인덴터별 확인)
MAX_DEPTH  = 2.5     # ★ 최대 눌림 깊이(mm) 마지노선 — 초과 금지(센서 보호)
Z_STEP     = 0.5     # z 계단 간격(mm)
XY_STEP    = 1.0     # xy 격자 간격(mm) — SOP 확정
XY_LIM     = 10.0    # ±10mm
U_WAIT     = 3.0     # 각 깊이 단계 측정 대기(U축)
DESCEND_V  = 0.6667  # 하강 속도
INDENTER   = "d5"    # 파일명 태그 (d5 / d10)
# ────────────────────────────────────────────────────────────────────────


def frange(start, end, step):
    res, cur = [], start
    while cur <= end + 1e-6:
        res.append(round(cur, 3))
        cur += step
    return res


def generate():
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"xy1_stepz_{INDENTER}_depth{MAX_DEPTH}mm.node"
    full_path = os.path.join(SAVE_DIR, fname)

    xs = frange(-XY_LIM, XY_LIM, XY_STEP)
    ys = frange(-XY_LIM, XY_LIM, XY_STEP)
    depths = frange(0.0, MAX_DEPTH, Z_STEP)   # [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert depths[-1] <= MAX_DEPTH + 1e-6, "깊이가 마지노선 초과"

    rows, order = [], 1

    def add(x, y, z, u, vs):
        nonlocal order
        rows.append([order, "4직선", x, y, round(z, 3), u, *vs, "NONE"])
        order += 1

    RAPID = [1, 10, 10, 10]
    DESC = [DESCEND_V] * 4
    HOLD = [1.0, 1.0, 1.0, 1.0]

    for y in ys:
        for x in xs:
            add(x, y, SAFE_Z, 0.0, RAPID)              # 안전높이로 쾌속 이동
            for d in depths:
                z = CONTACT_Z + d
                add(x, y, z, 0.0, DESC)                # 다음 깊이로 하강(계단)
                add(x, y, z, U_WAIT, HOLD)             # 정지·측정
                add(x, y, z, 0.0, RAPID)               # U 리셋
            add(x, y, SAFE_Z, 0.0, RAPID)              # 안전높이 복귀
    add(0.0, 0.0, 0.0, 0.0, RAPID)                     # 원점 복귀

    h1 = ["순서", "함수", "위치", "위치", "위치", "위치", "속도", "속도", "속도", "속도", "IO"]
    h2 = ["순서", "함수", "X", "Y", "Z", "U", "시작", "가속", "감속", "구동", "IO"]
    with open(full_path, "w", encoding="cp949") as f:
        f.write(",".join(h1) + "\n")
        f.write(",".join(h2) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    print("-" * 56)
    print("✅ xy1 계단식 z node 생성 완료")
    print(f"  파일: {full_path}")
    print(f"  격자: {len(xs)}×{len(ys)} = {len(xs)*len(ys)}점 (xy {XY_STEP}mm)")
    print(f"  깊이 계단: {depths} mm (최대 {MAX_DEPTH}mm)")
    print(f"  점당 측정 {len(depths)}회, 총 노드 {len(rows)}행")
    print(f"  ⚠️ SAFE_Z={SAFE_Z} CONTACT_Z={CONTACT_Z} — 실제 셋업 확인 필수")
    print("-" * 56)


if __name__ == "__main__":
    generate()
