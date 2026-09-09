# GPR Dash Application — Corrected Login SVG Package

## Login background correction
The login screen now renders the exact supplied `GPR_Drone_Live_Scan.svg` as a real HTML `<img>` element.

This is intentional: the previous implementation depended on CSS background layers and legacy PNG rules, which could prevent the SVG from appearing reliably.

### Exact asset
- `assets/GPR_Drone_Live_Scan.svg` — canonical supplied SVG used by the login screen.
- `assets/GPR_Live_PreLogin.svg` — synchronized copy of the same supplied SVG for compatibility with existing asset references.

### Implementation
`gpr_dash_app.py` now creates:
1. `login-background-svg` — actual SVG image element.
2. `login-background-overlay` — readability/dimming layer.
3. Existing glassmorphic login card above both layers.

`assets/style.css` contains a final high-specificity login block that disables the old login pseudo-element image layers and forces the SVG image to cover the full viewport.

## Other application behavior
No authenticated-workspace functionality, callbacks, model logic, upload handling, or post-login layout was intentionally changed.

## Verification
- Exact supplied SVG copied into the package.
- Python source compiles successfully.
- CSS contains no login dependency on the PNG background.
- ZIP contains the complete application folder and assets.

## Institutional branding update — 09 Sep 2026

- The authenticated application now displays the original AVNL logo sourced from the official AVNL website.
- The authenticated application now displays the exact CSIR-CMERI emblem supplied by the user as a local asset.
- The login SVG remains the supplied animated GPR drone scan.
- The visible `LIVE SCAN` badge has been removed from the embedded background artwork in `assets/GPR_Drone_Live_Scan.svg`; the drone, scan beam, radar animation, target pulse, and minimal status indicator remain.
- Branding is placed in the authenticated workspace header and sidebar; the login screen remains focused on secure access.

Brand sources:
- AVNL: https://avnl.co.in/
- AVNL OFMK: https://www.avnl.co.in/index.php/ordnance-factory-medak
- CSIR-CMERI: https://www.cmeri.res.in/


## Institutional branding
The authenticated application displays the original AVNL logo artwork and the exact supplied CSIR-CMERI emblem. The CMERI image is bundled locally at `assets/cmeri_cmeri_logo.png`, so it does not depend on an external image host.
