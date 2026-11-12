## Local patch to sportypy

sportypy is locally patched inside `.venv`: in
`sportypy/_base_classes/_base_feature.py`, the default `npoints` of
`create_circle` was lowered from 10000 to 200. At the original value,
`NHLRink.draw()` took ~40 s with matplotlib 3.11.x (about 1.3 million polygon
vertices, and matplotlib's `_update_patch_limits` walks them in pure Python).

If the venv is rebuilt or sportypy is reinstalled or upgraded, reapply:

    F=$(python -c "import sportypy._base_classes._base_feature as m; print(m.__file__)")
    sed -i 's/npoints = 10000/npoints = 200/' "$F"
    grep -n "npoints = " "$F"

Then `time python test.py` should show the rink drawing in about 2 s.

### Checking the current number of points

Show the patched default (run inside the venv):

    F=$(python -c "import sportypy._base_classes._base_feature as m; print(m.__file__)")
    grep -n "npoints = " "$F"

The `create_circle` line should show the patched value (not 10000).
