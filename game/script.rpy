init python:

    from enum import IntFlag
    from renpy.display.matrix import MatrixStack
    import math
    import random
    # =====================================================================
    # BREAKUP FOR REN'PY
    # =====================================================================
    #
    # Author: JAYBE (JAYBE GAMES)
    # Mail: jaybessecretservice@protonmail.com
    #
    # This script is based on renpy_breakup implementation from Neurochitin
    # which itself is closely based on the implementation in onscripter-ru
    # which is also based on Takashi Toyama's breakup.dll NScripter plugin
    # I know this script has a whole ass story.
    #
    # =====================================================================
    # HOW IT WORKS
    # =====================================================================
    #
    # Imagine the image as a sheet of paper made from lots of tiny tiles.
    # The effect's job is to make those tiles look as though they break
    # apart, move away from the original image, and disappear.
    #
    # There are two big moments in the life of an effect:
    #
    #   1. PRELOAD TIME
    #      We do the expensive setup ahead of time: resolve the image, render
    #      it into a texture, build the tile/grid data, create the meshes,
    #      and warm up the GPU/shader path.
    #
    #   2. PLAYBACK TIME
    #      breakup() simply takes that prepared package out of the cache.
    #      The CPU mostly feeds the GPU a small set of numbers describing
    #      'where the animation is right now'. The shader does the visual
    #      work for every vertex/pixel on the GPU.
    #
    # That separation is the main performance idea behind this version:
    # avoid rebuilding a complicated mesh during the first visible frame.
    # =====================================================================

    # -------------------------------------------------------------------------
    # BREAKUP CONSTANTS
    # -------------------------------------------------------------------------

    # These are expressed in effect-frames rather than screen FPS.
    # They define how quickly the dissolve/wipe/movement portions progress.
    BREAKUP_DISSOLVE_FRAMES = 1000
    BREAKUP_WIPE_FRAMES     = 3000
    BREAKUP_MOVE_FRAMES     = 850
    BREAKUP_TRAVEL_PX       = float(BREAKUP_MOVE_FRAMES)

    # Ren'Py/OpenGL meshes have a practical vertex limit, so large effects
    # are split into several Mesh2 objects instead of one giant mesh.
    BREAKUP_MAX_VERTICES    = 65535

    _sqrt = math.sqrt
    _cos = math.cos
    _sin = math.sin
    _atan2 = math.atan2
    _pi = math.pi
    _random = random.random

    _INV_DISSOLVE = 1.0 / float(BREAKUP_DISSOLVE_FRAMES)
    _INV_MOVE = 1.0 / float(BREAKUP_MOVE_FRAMES)
    _INV_180_PI = 180.0 / _pi

    # Trig is used a lot when we choose random-looking movement directions.
    # Precomputing 360 sine/cosine values avoids repeating the math later.
    _COS_LIST = [_cos(i * _pi / 180.0) for i in range(360)]
    _SIN_LIST = [_sin(i * _pi / 180.0) for i in range(360)]

    _ANGLE_STEP = 0.015707963267948967
    _CIRC_CONST = 5.026548245743669

    # For each possible circle segment count, remember the unit-circle
    # points. This is used by the original geometry path when building
    # fragment outlines.
    _CIRCLE_LOOKUP = [None] * 100
    for segs in range(2, 100):
        _CIRCLE_LOOKUP[segs] = [
            (_cos(2.0 * _pi * i / segs), _sin(2.0 * _pi * i / segs))
            for i in range(segs)
        ]

    # -------------------------------------------------------------------------
    # GPU SHADER:
    # -------------------------------------------------------------------------
    # Python creates the 'skeleton' of the mesh once. The shader then makes
    # that skeleton look different from frame to frame.
    #
    # A useful mental model is:
    #     Python = builds the Lego pieces.
    #     GPU     = decides where those pieces are RIGHT NOW.
    #
    # -------------------------------------------------------------------------
    #
    # Python builds the static topology once.
    # The GPU evaluates time-dependent position, radius and visibility.
    #
    # This removes the per-frame Python mesh construction and buffer uploads
    # while preserving the existing timing equations and random setup.
    # -------------------------------------------------------------------------

    # Register one custom shader program with Ren'Py under the name
    # 'breakup.gpu'. The Displayable later attaches this shader to each
    # mesh it draws.
    renpy.register_shader(
        "breakup.gpu",
        variables="""
            uniform vec4 u_breakup_params;
            uniform vec2 u_breakup_inv_size;
            uniform float u_breakup_max_diag;
            uniform float u_breakup_radius;
            uniform float u_breakup_max_segments;

            attribute vec2 a_breakup_move;
            attribute vec2 a_breakup_offset;
            attribute float a_breakup_state;
            attribute float a_breakup_diag;
            attribute float a_breakup_kind;
            attribute float a_breakup_segment;
            attribute float a_breakup_point;
            attribute float a_breakup_closing;

            varying float v_breakup_visible;
        """,
        vertex_300="""
            float breakup_frame = u_breakup_params.x;
            float breakup_dissolve = u_breakup_params.y;
            float breakup_move = u_breakup_params.z;
            float breakup_cutoff = u_breakup_params.w;
            float breakup_remaining = a_breakup_state - breakup_frame;
            float breakup_resize = 1.0;
            if (breakup_remaining < breakup_dissolve) {
                breakup_resize = breakup_remaining / breakup_dissolve;
            }

            bool breakup_lowpoly = breakup_resize < 0.15;
            float breakup_visible = 1.0;

            if (a_breakup_kind < 0.5) {
                if (a_breakup_diag > u_breakup_max_diag) {
                    breakup_visible = 0.0;
                }

                if (breakup_remaining <= breakup_cutoff) {
                    breakup_visible = 0.0;
                }

                if (breakup_lowpoly) {
                    breakup_visible = 0.0;
                }

                float breakup_radius = u_breakup_radius * breakup_resize;
                if (breakup_radius <= 0.0) {
                    breakup_visible = 0.0;
                }

                float breakup_segments = floor(
                    5.026548245743669 * sqrt(max(0.0, breakup_radius))
                ) + 1.0;

                if (a_breakup_closing > 0.5) {
                    if (abs(a_breakup_segment - breakup_segments) > 0.01) {
                        breakup_visible = 0.0;
                    }
                }
                else if (a_breakup_segment >= breakup_segments - 0.01) {
                    breakup_visible = 0.0;
                }
            }
            else if (a_breakup_kind < 1.5) {
                if (a_breakup_diag > u_breakup_max_diag) {
                    breakup_visible = 0.0;
                }

                if (breakup_remaining <= breakup_cutoff) {
                    breakup_visible = 0.0;
                }

                if (!breakup_lowpoly) {
                    breakup_visible = 0.0;
                }
            }
            else {
                if (abs(a_breakup_diag - u_breakup_max_diag) > 0.01) {
                    breakup_visible = 0.0;
                }
            }

            float breakup_move_t = 0.0;
            if (breakup_remaining < breakup_move) {
                breakup_move_t = (breakup_move - breakup_remaining) / breakup_move;
                breakup_move_t = breakup_move_t * breakup_move_t *
                                (3.0 - 2.0 * breakup_move_t);
            }

            vec2 breakup_position = a_position.xy;
            vec2 breakup_offset = vec2(0.0, 0.0);

            if (a_breakup_kind < 0.5) {
                float breakup_segments = floor(
                    5.026548245743669 *
                    sqrt(max(0.0, u_breakup_radius * breakup_resize))
                ) + 1.0;

                if (a_breakup_point > -0.5) {
                    float breakup_point = a_breakup_point;
                    float breakup_angle =
                        6.283185307179586 * breakup_point / breakup_segments;

                    breakup_offset = u_breakup_radius *
                        vec2(cos(breakup_angle), sin(breakup_angle));
                }
                else if (a_breakup_point < -1.5) {
                    float breakup_point = breakup_segments - 1.0;
                    float breakup_angle =
                        6.283185307179586 * breakup_point / breakup_segments;

                    breakup_offset = u_breakup_radius *
                        vec2(cos(breakup_angle), sin(breakup_angle));
                }
            }
            else if (a_breakup_kind < 1.5) {
                breakup_offset = a_breakup_offset;
            }

            if (a_breakup_kind < 1.5) {
                breakup_position += a_breakup_move * breakup_move_t;
                breakup_position += breakup_offset * breakup_resize;
            }

            gl_Position = u_transform * vec4(
                breakup_position.x,
                breakup_position.y,
                0.0,
                1.0
            );


            v_breakup_visible = breakup_visible;

            if (a_breakup_kind < 0.5) {
                v_tex_coord = a_tex_coord +
                    breakup_offset * breakup_resize * u_breakup_inv_size;
            }
            else if (a_breakup_kind < 1.5) {
                v_tex_coord = a_tex_coord +
                    breakup_offset * u_breakup_inv_size;
            }
        """,
        fragment_150="""
            if (v_breakup_visible < 0.5) {
                discard;
            }
        """,
    )


    # -------------------------------------------------------------------------
    # MESH ATTRIBUTE LAYOUT
    # -------------------------------------------------------------------------
    # This tells Mesh2 how the packed attribute array is laid out. In other
    # words, when Python says 'here are the numbers for a vertex', this list
    # tells the GPU what each number means.
    # A separate attribute layout lets the mesh keep the static data on the
    # GPU. Mesh2 itself is Cython-backed in current Ren'Py releases.
    _BREAKUP_LAYOUT = renpy.gl2.gl2mesh.AttributeLayout()
    _BREAKUP_LAYOUT.add_attribute("a_tex_coord", 2)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_move", 2)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_offset", 2)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_state", 1)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_diag", 1)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_kind", 1)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_segment", 1)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_point", 1)
    _BREAKUP_LAYOUT.add_attribute("a_breakup_closing", 1)

    # Effect direction flags. IntFlag lets several options be combined, e.g.
    # LEFT | JUMBLE.
    class BreakupMode(IntFlag):
        LEFT = 1
        LOWER = 2
        JUMBLE = 4


    # One BreakupCell represents one small tile in the image grid.
    # Most of the values here are calculated once during preload and then
    # reused during playback.
    class BreakupCell(python_object):
        __slots__ = (
            "grid_x", "grid_y",
            "base_x", "base_y",
            "move_x", "move_y",
            "move_px_x", "move_px_y",
            "state", "diagonal",
            "uv_s", "uv_t"
        )

        def __init__(self):
            self.grid_x = 0
            self.grid_y = 0
            self.base_x = 0.0
            self.base_y = 0.0
            self.move_x = 0.0
            self.move_y = 0.0
            self.move_px_x = 0.0
            self.move_px_y = 0.0
            self.state = 0.0
            self.diagonal = 0
            self.uv_s = 0.0
            self.uv_t = 0.0


    # The grid is the complete precomputed map of the image.
    # Besides holding all cells, it remembers diagonal timing information
    # and the finished static Mesh2 objects.
    class BreakupGrid(python_object):
        __slots__ = (
            "cells", "cell_size", "cells_x", "cells_y",
            "diag_slices", "diag_min_state",
            "mode", "total_frames", "diag_count",
            "mesh_groups", "inv_iw", "inv_ih",
            "max_segments", "radius"
        )

        def __init__(self):
            self.cells = python_list()
            self.cell_size = 0
            self.cells_x = 0
            self.cells_y = 0
            self.diag_slices = python_list()
            self.diag_min_state = python_list()
            self.mode = None
            self.total_frames = 0.0
            self.diag_count = 0

            self.mesh_groups = python_list()
            self.inv_iw = 0.0
            self.inv_ih = 0.0
            self.max_segments = 2
            self.radius = 0.0


    # -------------------------------------------------------------------------
    # STATIC MESH BUILDER
    # -------------------------------------------------------------------------
    # The functions below turn the grid into GPU-ready triangles.
    # They are intentionally called during preload, not every animation frame.
    # A temporary Python-side container used while assembling one Mesh2.
    # Once filled, it gets converted to the actual Ren'Py/OpenGL mesh.
    class _BreakupMeshChunk(python_object):
        __slots__ = (
            "vertices",
            "attributes",
            "indices",
            "vertex_count",
            "triangle_count",
        )

        def __init__(self):
            self.vertices = python_list()
            self.attributes = python_list()
            self.indices = python_list()
            self.vertex_count = 0
            self.triangle_count = 0


    # Add one vertex and all of its extra breakup data to the current chunk.
    # The important part is that position, UVs, movement, timing, and geometry
    # type are stored together so the shader can read them later.
    def _breakup_append_vertex(
        chunk,
        x,
        y,
        move_x,
        move_y,
        offset_x,
        offset_y,
        uv_s,
        uv_t,
        state,
        diag,
        kind,
        segment,
        point,
        closing,
    ):
        i = chunk.vertex_count

        chunk.vertices.append(x)
        chunk.vertices.append(y)

        a = chunk.attributes
        a.append(uv_s)
        a.append(uv_t)

        a.append(move_x)
        a.append(move_y)

        a.append(offset_x)
        a.append(offset_y)

        a.append(state)
        a.append(float(diag))
        a.append(kind)
        a.append(segment)
        a.append(point)
        a.append(closing)

        chunk.vertex_count = i + 1
        return i


    # A triangle is simply three vertices plus one entry in the index buffer.
    # This helper keeps all of the repetitive packing code in one place.
    def _breakup_append_triangle(
        chunk,
        x1, y1, x2, y2, x3, y3,
        move_x, move_y,
        ox1, oy1, ox2, oy2, ox3, oy3,
        uv_s1, uv_t1, uv_s2, uv_t2, uv_s3, uv_t3,
        state, diag, kind, segment,
        point1, point2, point3,
        closing,
    ):
        p0 = _breakup_append_vertex(
            chunk,
            x1, y1,
            move_x, move_y,
            ox1, oy1,
            uv_s1, uv_t1,
            state, diag, kind, segment, point1, closing,
        )
        p1 = _breakup_append_vertex(
            chunk,
            x2, y2,
            move_x, move_y,
            ox2, oy2,
            uv_s2, uv_t2,
            state, diag, kind, segment, point2, closing,
        )
        p2 = _breakup_append_vertex(
            chunk,
            x3, y3,
            move_x, move_y,
            ox3, oy3,
            uv_s3, uv_t3,
            state, diag, kind, segment, point3, closing,
        )

        chunk.indices.append(p0)
        chunk.indices.append(p1)
        chunk.indices.append(p2)
        chunk.triangle_count += 1


    # Convert our Python lists into Ren'Py's GPU-backed Mesh2 object.
    # After this point the mesh can be handed to the renderer.
    def _breakup_make_mesh(chunk):
        mesh = renpy.gl2.gl2mesh2.Mesh2(
            _BREAKUP_LAYOUT,
            chunk.vertex_count,
            chunk.triangle_count,
        )

        mesh.set_geometry_data(chunk.vertices)
        mesh.set_attribute_data(chunk.attributes)
        mesh.set_triangle_data(chunk.indices)

        return mesh


    # The intact part of the image needs only ordinary triangles. It does not
    # need the fragment-movement attributes, so those values are zeroed out.
    def _breakup_add_region_triangle(chunk, c1, c2, c3, diag):
        _breakup_append_triangle(
            chunk,
            c1.base_x, c1.base_y,
            c2.base_x, c2.base_y,
            c3.base_x, c3.base_y,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            c1.uv_s, c1.uv_t,
            c2.uv_s, c2.uv_t,
            c3.uv_s, c3.uv_t,
            0.0,
            diag,
            2.0,
            0.0,
            -1.0, -1.0, -1.0,
            0.0,
        )


    # Build every static mesh used by this effect.
    #
    # For each cell we store enough geometry for the largest supported circle.
    # The shader later hides the unused segments. This is the key trick that
    # lets the topology stay fixed while radius/shrink changes over time.
    def _breakup_build_static_meshes(g):
        cells = g.cells
        max_segments = g.max_segments
        radius = g.radius

        groups = [_BreakupMeshChunk()]

        # We deliberately build the maximum-resolution version once. Think of
        # this as preparing the biggest possible set of puzzle pieces so the
        # GPU can simply hide the pieces it does not need at a given moment.
        # The topology is deliberately generated at the largest segment count.
        # GLSL selects the exact segment count at runtime.
        for cell in cells:
            chunk = groups[-1]

            # Check whether adding this cell would exceed the mesh's vertex
            # budget. If so, start another chunk.
            highpoly_triangles = (max_segments - 1) + (max_segments - 1)
            required_highpoly_vertices = highpoly_triangles * 3
            required_lowpoly_vertices = 6
            required_vertices = (
                required_highpoly_vertices +
                required_lowpoly_vertices
            )

            if (
                chunk.vertex_count + required_vertices >
                BREAKUP_MAX_VERTICES
            ):
                chunk = _BreakupMeshChunk()
                groups.append(chunk)

            # High-poly fan: many triangles radiate from the cell center.
            # Each triangle is tagged with its segment number, allowing the
            # shader to enable/disable segments without rebuilding the mesh.
            # High-poly fan triangles. Each triangle carries a constant segment
            # selector, so the GPU can discard complete triangles without
            # interpolation artifacts.
            for seg in range(1, max_segments):
                _breakup_append_triangle(
                    chunk,
                    cell.base_x, cell.base_y,
                    cell.base_x, cell.base_y,
                    cell.base_x, cell.base_y,
                    cell.move_px_x, cell.move_px_y,
                    0.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0,
                    cell.uv_s, cell.uv_t,
                    cell.uv_s, cell.uv_t,
                    cell.uv_s, cell.uv_t,
                    cell.state,
                    cell.diagonal,
                    0.0,
                    float(seg),
                    -1.0,
                    float(seg - 1),
                    float(seg),
                    0.0,
                )

            # One extra 'closing' triangle is kept for each possible segment
            # count. Together with the fan above, this lets the shader switch
            # cleanly between different circle resolutions.
            # One closing triangle is stored for every possible segment count.
            # The shader activates exactly the one matching the current radius.
            for seg in range(2, max_segments + 1):
                _breakup_append_triangle(
                    chunk,
                    cell.base_x, cell.base_y,
                    cell.base_x, cell.base_y,
                    cell.base_x, cell.base_y,
                    cell.move_px_x, cell.move_px_y,
                    0.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0,
                    cell.uv_s, cell.uv_t,
                    cell.uv_s, cell.uv_t,
                    cell.uv_s, cell.uv_t,
                    cell.state,
                    cell.diagonal,
                    0.0,
                    float(seg),
                    -1.0,
                    float(seg - 1),
                    0.0,
                    1.0,
                )

            # When a fragment is tiny, the original effect uses a cheap
            # four-corner shape instead of a detailed circle.
            # Low-poly fallback. This preserves the intended four-corner
            # topology used by the original low-poly path.
            r = radius
            lowpoly = (
                (r, r),
                (-r, -r),
                (-r, r),
                (r, -r),
            )

            _breakup_append_triangle(
                chunk,
                cell.base_x, cell.base_y,
                cell.base_x, cell.base_y,
                cell.base_x, cell.base_y,
                cell.move_px_x, cell.move_px_y,
                lowpoly[0][0], lowpoly[0][1],
                lowpoly[1][0], lowpoly[1][1],
                lowpoly[2][0], lowpoly[2][1],
                cell.uv_s, cell.uv_t,
                cell.uv_s, cell.uv_t,
                cell.uv_s, cell.uv_t,
                cell.state,
                cell.diagonal,
                1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
            )

            _breakup_append_triangle(
                chunk,
                cell.base_x, cell.base_y,
                cell.base_x, cell.base_y,
                cell.base_x, cell.base_y,
                cell.move_px_x, cell.move_px_y,
                lowpoly[0][0], lowpoly[0][1],
                lowpoly[2][0], lowpoly[2][1],
                lowpoly[3][0], lowpoly[3][1],
                cell.uv_s, cell.uv_t,
                cell.uv_s, cell.uv_t,
                cell.uv_s, cell.uv_t,
                cell.state,
                cell.diagonal,
                1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
            )

        # These triangles describe the part of the picture that has NOT broken
        # yet. Keeping this as static geometry means the CPU does not have to
        # reconstruct the intact region during every frame.
        # Static unbroken regions. There are only a handful of triangles per
        # diagonal, so keeping all diagonal variants is inexpensive.
        region_groups = [_BreakupMeshChunk()]

        for max_active_diag in range(g.diag_count - 1):
            first = g.diag_slices[max_active_diag][0]
            next_start = g.diag_slices[max_active_diag + 1][0]
            last = next_start - 1

            if last < first:
                continue

            chunk = region_groups[-1]
            if chunk.vertex_count + 6 > BREAKUP_MAX_VERTICES:
                chunk = _BreakupMeshChunk()
                region_groups.append(chunk)

            last_cell = cells[g.cells_x * g.cells_y - 1]

            for idx in (first, last):
                cell = cells[idx]

                if (
                    cell.grid_x != last_cell.grid_x
                    and cell.grid_y != last_cell.grid_y
                ):
                    if (
                        cell.grid_x == 0
                        or cell.grid_x == g.cells_x - 1
                    ):
                        shared_x = cell.grid_x
                        shared_y = last_cell.grid_y
                    else:
                        shared_x = last_cell.grid_x
                        shared_y = cell.grid_y

                    shared_base_x = float(shared_x * g.cell_size)
                    shared_base_y = float(shared_y * g.cell_size)
                    shared_uv_s = shared_base_x * g.inv_iw
                    shared_uv_t = shared_base_y * g.inv_ih

                    _breakup_append_triangle(
                        chunk,
                        cell.base_x, cell.base_y,
                        last_cell.base_x, last_cell.base_y,
                        shared_base_x, shared_base_y,
                        0.0, 0.0,
                        0.0, 0.0,
                        0.0, 0.0,
                        0.0, 0.0,
                        cell.uv_s, cell.uv_t,
                        last_cell.uv_s, last_cell.uv_t,
                        shared_uv_s, shared_uv_t,
                        0.0,
                        max_active_diag,
                        2.0,
                        0.0,
                        -1.0, -1.0, -1.0,
                        0.0,
                    )

            _breakup_add_region_triangle(
                chunk,
                cells[last],
                cells[first],
                last_cell,
                max_active_diag,
            )

        mesh_groups = python_list()

        # Match the original draw order:
        # unbroken region first, fragment geometry second.
        for chunk in region_groups:
            if chunk.triangle_count:
                mesh_groups.append(_breakup_make_mesh(chunk))

        for chunk in groups:
            if chunk.triangle_count:
                mesh_groups.append(_breakup_make_mesh(chunk))

        g.mesh_groups = mesh_groups


    # =====================================================================
    # PRELOAD CACHE
    # =====================================================================
    #
    # This dictionary is basically a shelf of prepared effects.
    # A cache key says exactly which image/settings the package belongs to.
    #
    # Example idea:
    #     ('ImageName', mode=0, separation=16, radius=12)
    #
    # breakup() must ask for the exact same combination. That strict matching
    # is intentional: it prevents accidentally using geometry prepared for a
    # different visual configuration.
    # =====================================================================
    # -------------------------------------------------------------------------
    # PRELOAD CACHE
    #
    # The preload contains the actual data that BreakupDisplayable needs:
    #   * the resolved source displayable;
    #   * its rendered GL texture;
    #   * the fully-generated breakup grid;
    #   * the static Mesh2 objects.
    #
    # IMPORTANT: this cache is persistent. breakup() BORROWS an entry instead
    # of consuming it, so the same preloaded image can be used again later
    # without another expensive preparation step. The entry stays here until
    # breakup_clear(...) or breakup_clear_all() explicitly removes it.
    #
    # Think of this as a small library of ready-to-play effects:
    #
    #     breakup_preload("bgMetaCorridor")
    #              |
    #              v
    #       [ready in memory]
    #              |
    #        +-----+-----+
    #        |           |
    #     breakup()   breakup()
    #        |           |
    #        +-----+-----+
    #              |
    #        breakup_clear()
    #
    # Keeping entries around uses memory (and can use VRAM through the cached
    # source texture), so long-lived caches should be cleared when a scene,
    # chapter, or group of effects is no longer needed.
    # -------------------------------------------------------------------------

    _BREAKUP_PRELOAD_CACHE = {}
    # Set False after diagnosing if you want quiet logs.
    _BREAKUP_DEBUG = True


    # Centralized debug logger. This is useful when diagnosing a freeze because
    # every expensive stage can report how long it took and what object it used.
    def _breakup_debug(message):
        if not _BREAKUP_DEBUG:
            return

        try:
            import time as _breakup_time
            stamp = _breakup_time.perf_counter()
        except Exception:
            stamp = 0.0

        line = "[Breakup %.6f] %s" % (stamp, message)
        print(line)
        try:
            renpy.log(line)
        except Exception:
            pass


    # Make a stable cache key. Strings are keyed by their name; other objects
    # are keyed by their identity so we do not silently mix unrelated objects.
    def _breakup_preload_key(image, mode, separation, radius):
        if isinstance(image, str):
            image_key = ("name", image)
        else:
            image_key = ("object", id(image))

        return (
            image_key,
            int(mode),
            int(separation),
            float(radius),
        )


    # Build exactly the same render tree that the live Displayable uses.
    # The difference is that preload calls it directly so the render/shader
    # path gets touched before the player ever sees the real animation.
    def _breakup_make_render_from_source(
        prepared,
        source,
        factor,
    ):
        """
        Builds the same render tree used by BreakupDisplayable.render().

        `source` may be either the source Render or the preloaded GL texture.
        """
        c_w = int(prepared.width)
        c_h = int(prepared.height)

        mesh_groups, params, inv_size, max_diag = (
            prepared._build_meshes(float(factor))
        )

        total = renpy.Render(c_w, c_h)

        for mesh in mesh_groups:
            sub = renpy.Render(c_w, c_h)
            sub.mesh = mesh
            sub.add_shader("renpy.texture")
            sub.add_shader("breakup.gpu")
            sub.add_uniform("u_breakup_params", params)
            sub.add_uniform("u_breakup_inv_size", inv_size)
            sub.add_uniform("u_breakup_max_diag", max_diag)
            sub.add_uniform(
                "u_breakup_radius",
                float(prepared.radius),
            )
            sub.add_uniform(
                "u_breakup_max_segments",
                float(prepared.grid.max_segments),
            )
            sub.blit(source, (0, 0))
            total.blit(sub, (0, 0))

        return total


    # Prepare ONE image completely. This is where most of the expensive work
    # has deliberately been moved out of the visible animation's first frame.
    def _breakup_prepare_one(image, mode, separation, radius):
        import time as _breakup_time

        t0 = _breakup_time.perf_counter()
        _breakup_debug(
            "PREPARE START image=%r id=%s mode=%s separation=%s radius=%s"
            % (image, id(image), int(mode), separation, radius)
        )

        draw = getattr(renpy.display, "draw", None)
        if draw is None:
            raise RuntimeError(
                "breakup_preload(): Ren'Py renderer is not initialized yet."
            )

        # Resolve the image once. Keeping this exact displayable object avoids
        # doing another lookup when the cached package is consumed later.
        # Resolve the image exactly once and keep this exact displayable object.
        widget = renpy.displayable(image)
        _breakup_debug(
            "PREPARE RESOLVED input_id=%s widget_id=%s widget_type=%s"
            % (id(image), id(widget), type(widget).__name__)
        )

        # Render the source at the actual game screen size so we get the same
        # dimensions/texture that the live effect expects.
        screen_w = int(config.screen_width)
        screen_h = int(config.screen_height)
        _breakup_debug(
            "PREPARE SOURCE RENDER start screen=%sx%s" % (screen_w, screen_h)
        )

        source_t0 = _breakup_time.perf_counter()
        source_render = renpy.render(widget, screen_w, screen_h, 0.0, 0.0)
        source_dt = _breakup_time.perf_counter() - source_t0
        width, height = source_render.get_size()
        _breakup_debug(
            "PREPARE SOURCE RENDER done dt=%.6fs size=%sx%s render_id=%s"
            % (source_dt, width, height, id(source_render))
        )

        # Turn that render into a GPU texture. The live effect will sample this
        # texture instead of asking Ren'Py to resolve/render the source again.
        _breakup_debug("PREPARE SOURCE TEXTURE start")
        tex_t0 = _breakup_time.perf_counter()
        source_texture = draw.render_to_texture(source_render, properties={})
        tex_dt = _breakup_time.perf_counter() - tex_t0
        _breakup_debug(
            "PREPARE SOURCE TEXTURE done dt=%.6fs texture_id=%s texture_type=%s"
            % (tex_dt, id(source_texture), type(source_texture).__name__)
        )

        # Create an internal helper Displayable purely as a convenient holder for
        # the grid-building logic. `_breakup_preparing` marks it as preload-only.
        prepared = BreakupDisplayable(
            widget,
            delay=1.0,
            reverse=False,
            mode=mode,
            separation=int(separation),
            radius=float(radius),
            _breakup_preparing=True,
        )

        prepared.width = int(width)
        prepared.height = int(height)

        _breakup_debug(
            "PREPARE GRID start width=%s height=%s" % (width, height)
        )
        # Create the grid and static meshes now, while we are still in preload.
        # This is the expensive CPU-side preparation we do not want on the
        # first visible animation frame.
        grid_t0 = _breakup_time.perf_counter()
        prepared._ensure_grid(width, height)
        prepared._setup_cells_once()
        grid_dt = _breakup_time.perf_counter() - grid_t0
        _breakup_debug(
            "PREPARE GRID done dt=%.6fs cells=%s meshes=%s cells_x=%s cells_y=%s max_segments=%s"
            % (
                grid_dt,
                len(prepared.grid.cells),
                len(prepared.grid.mesh_groups),
                prepared.grid.cells_x,
                prepared.grid.cells_y,
                prepared.grid.max_segments,
            )
        )

        # Make sure the shader program is present in Ren'Py's shader cache.
        shader_cache = getattr(draw, "shader_cache", None)
        if shader_cache is None:
            raise RuntimeError(
                "breakup_preload(): Ren'Py shader cache is unavailable; cannot verify GPU preparation."
            )

        _breakup_debug("PREPARE SHADER CACHE lookup start")
        shader_t0 = _breakup_time.perf_counter()
        shader_cache.get(("renpy.texture", "breakup.gpu"))
        shader_dt = _breakup_time.perf_counter() - shader_t0
        _breakup_debug(
            "PREPARE SHADER CACHE lookup done dt=%.6fs" % shader_dt
        )

        # GPU warmup: actually submit a representative breakup render path to
        # the renderer. Merely creating Python objects is not enough to guarantee
        # that all GPU-side work has already been touched.
        _breakup_debug("PREPARE GPU WARMUP start")
        warm_t0 = _breakup_time.perf_counter()
        warm_render = _breakup_make_render_from_source(
            prepared,
            source_texture,
            0.0,
        )

        identity = MatrixStack([1, 0, 0, 1])
        draw.load_all_textures(warm_render, identity)

        is_pixel_opaque = getattr(draw, "is_pixel_opaque", None)
        if is_pixel_opaque is None:
            raise RuntimeError(
                "breakup_preload(): Ren'Py GL2 draw.is_pixel_opaque() is unavailable."
            )

        # Run two renderer checks on purpose. This is intended as a real warmup,
        # not just a Python-level 'pretend' render.
        # Two actual renderer passes: this is deliberately not a Python-only
        # render-tree warmup.
        is_pixel_opaque(warm_render)
        is_pixel_opaque(warm_render)

        warm_dt = _breakup_time.perf_counter() - warm_t0
        _breakup_debug(
            "PREPARE GPU WARMUP done dt=%.6fs warm_render_id=%s"
            % (warm_dt, id(warm_render))
        )

        total_dt = _breakup_time.perf_counter() - t0
        _breakup_debug(
            "PREPARE DONE total=%.6fs image=%r widget_id=%s texture_id=%s grid_id=%s meshes=%s"
            % (
                total_dt,
                image,
                id(widget),
                id(source_texture),
                id(prepared.grid),
                len(prepared.grid.mesh_groups),
            )
        )

        return (
            widget,
            int(width),
            int(height),
            prepared.grid,
            source_texture,
        )


    # Public preload function called by the game. You can pass several images
    # at once; each one gets its own cache entry.
    def breakup_preload(
        *images,
        mode=BreakupMode.LEFT,
        separation=16,
        radius=12.0,
        fps=None,
    ):
        # IMPORTANT: preload settings must match breakup() exactly.
        # The cache does not guess or silently rebuild a different configuration.
        """
        Strict breakup preload. Every user-facing breakup() call must reuse
        an entry with the exact same image, mode, separation, and radius.

        Example for: breakup("ImageName", 7, True, 0)
            breakup_preload("ImageName", mode=0)

        There is intentionally no lazy fallback.
        """
        import time as _breakup_time
        t0 = _breakup_time.perf_counter()

        # No images means there is nothing to prepare, so leave immediately.
        if not images:
            _breakup_debug("PRELOAD called with zero images -> nothing to do")
            return 0

        _breakup_debug(
            "PRELOAD START count=%s images=%r mode=%s separation=%s radius=%s fps=%r cache_size_before=%s"
            % (len(images), images, int(mode), separation, radius, fps, len(_BREAKUP_PRELOAD_CACHE))
        )

        # Give the loading screen a hard frame before doing the blocking setup.
        # This does not make the work magically asynchronous; it simply gives
        # Ren'Py a chance to display that loading/black frame first.
        renpy.pause(0.01, hard=True)
        _breakup_debug("PRELOAD loading screen yielded one hard frame")

        count = 0

        # Prepare each requested image one at a time and store the finished
        # package under the exact settings used for this preload call.
        for index, image in enumerate(images):
            key = _breakup_preload_key(
                image, mode, separation, radius
            )

            _breakup_debug(
                "PRELOAD[%s/%s] image=%r id=%s key=%r cache_contains=%s"
                % (index + 1, len(images), image, id(image), key, key in _BREAKUP_PRELOAD_CACHE)
            )

            # Do not pay the preparation cost twice for the same key.
            if key in _BREAKUP_PRELOAD_CACHE:
                _breakup_debug(
                    "PRELOAD[%s/%s] SKIP already cached"
                    % (index + 1, len(images))
                )
                continue

            # This call performs the expensive source-render, texture, grid,
            # mesh, shader-cache, and GPU-warmup steps.
            entry = _breakup_prepare_one(
                image, mode, int(separation), float(radius)
            )

            # Store the finished package. From here on, breakup() can just
            # retrieve it instead of rebuilding everything.
            _BREAKUP_PRELOAD_CACHE[key] = entry
            count += 1

            _breakup_debug(
                "PRELOAD[%s/%s] STORED key=%r cache_size=%s"
                % (index + 1, len(images), key, len(_BREAKUP_PRELOAD_CACHE))
            )

        _breakup_debug(
            "PRELOAD DONE prepared=%s total_dt=%.6fs cache_size_after=%s keys=%r"
            % (
                count,
                _breakup_time.perf_counter() - t0,
                len(_BREAKUP_PRELOAD_CACHE),
                list(_BREAKUP_PRELOAD_CACHE.keys()),
            )
        )
        return count


    # Fetch the prepared package for a live breakup() call.
    # There is an important Ren'Py wrinkle here: the engine may evaluate
    # displayables during prediction BEFORE it actually shows them.
    def _breakup_take_preload(image, mode, separation, radius):
        key = _breakup_preload_key(
            image,
            mode,
            separation,
            radius,
        )

        # Prediction is Ren'Py checking ahead of time. Think of it like the
        # engine asking 'what would I draw here later?' without actually playing
        # the scene yet.
        predicting = bool(getattr(renpy.display.predict, "predicting", False))

        _breakup_debug(
            "CACHE LOOKUP key=%r image=%r id=%s mode=%s separation=%s radius=%s "
            "cache_size=%s predicting=%s"
            % (
                key, image, id(image), int(mode), separation, radius,
                len(_BREAKUP_PRELOAD_CACHE), predicting,
            )
        )

        # During prediction we LOOK at the cache but do not remove anything.
        # Otherwise the prediction pass would eat the one-shot package and the
        # real animation would immediately crash with a cache miss.
        # Ren'Py evaluates `show expression ...` during prediction and then
        # evaluates it again when the statement is actually executed. During
        # prediction we MUST NOT consume the one-shot preload entry, or the
        # real execution will see a cache miss.
        if predicting:
            prepared = _BREAKUP_PRELOAD_CACHE.get(key)
            if prepared is None:
                same_image_keys = [
                    k for k in _BREAKUP_PRELOAD_CACHE.keys()
                    if k[0] == key[0]
                    and k[2] == key[2]
                    and k[3] == key[3]
                ]

                    # If the image exists but the mode/settings differ, this
                    # message makes that mistake obvious in the debug log.
                if same_image_keys:
                    _breakup_debug(
                        "CACHE PREDICTION MISS DUE TO MODE image=%r requested_mode=%s "
                        "requested_key=%r available_same_image_keys=%r"
                        % (image, int(mode), key, same_image_keys)
                    )
                else:
                    _breakup_debug(
                        "CACHE PREDICTION MISS image=%r requested_key=%r remaining_keys=%r"
                        % (image, key, list(_BREAKUP_PRELOAD_CACHE.keys()))
                    )

                raise RuntimeError(
                    "breakup(): no exact preload exists for image %r with mode=%r, "
                    "separation=%r, radius=%r during prediction. "
                    "Call breakup_preload() first with the exact same settings."
                    % (image, mode, separation, radius)
                )

            _breakup_debug(
                "CACHE PREDICTION HIT/KEEP image=%r id=%s key=%r "
                "prepared_widget_id=%s texture_id=%s grid_id=%s mesh_count=%s cache_size_now=%s"
                % (
                    image, id(image), key,
                    id(prepared[0]), id(prepared[4]), id(prepared[3]),
                    len(prepared[3].mesh_groups), len(_BREAKUP_PRELOAD_CACHE),
                )
            )
            return prepared

        # This is the real, visible execution path.
        #
        # The old version used `pop()` here. That meant the first real breakup()
        # call took the entry out of the cache and the preload could only be used
        # once.
        #
        # The persistent-cache version deliberately uses `get()` instead. The
        # live animation receives references to the prepared objects, while the
        # cache keeps its own references too. That means a second breakup() call
        # can reuse the exact same prepared texture/grid/meshes.
        #
        # In other words: the cache OWNS the prepared package; the animation
        # temporarily BORROWS it. The package remains available until the user
        # explicitly clears it.
        prepared = _BREAKUP_PRELOAD_CACHE.get(key)
        if prepared is None:
            same_image_keys = [
                k for k in _BREAKUP_PRELOAD_CACHE.keys()
                if k[0] == key[0]
                and k[2] == key[2]
                and k[3] == key[3]
            ]
                # Again, distinguish 'wrong settings' from 'nothing was cached'
                # in the diagnostic output.
            if same_image_keys:
                _breakup_debug(
                    "CACHE MISS DUE TO MODE image=%r requested_mode=%s requested_key=%r "
                    "available_same_image_keys=%r"
                    % (image, int(mode), key, same_image_keys)
                )
            else:
                _breakup_debug(
                    "CACHE MISS image=%r requested_key=%r remaining_keys=%r"
                    % (image, key, list(_BREAKUP_PRELOAD_CACHE.keys()))
                )

            raise RuntimeError(
                "breakup(): no exact preload exists for image %r with mode=%r, "
                "separation=%r, radius=%r. "
                "Available matching image entries: %r. "
                "The preload must use the exact same mode/settings as breakup()."
                % (
                    image,
                    mode,
                    separation,
                    radius,
                    same_image_keys,
                )
            )

        _breakup_debug(
            "CACHE HIT/KEEP image=%r id=%s key=%r prepared_widget_id=%s texture_id=%s grid_id=%s mesh_count=%s"
            % (
                image,
                id(image),
                key,
                id(prepared[0]),
                id(prepared[4]),
                id(prepared[3]),
                len(prepared[3].mesh_groups),
            )
        )
        _breakup_debug(
            "CACHE KEPT cache_size_now=%s -> reusable until explicit clear"
            % len(_BREAKUP_PRELOAD_CACHE)
        )
        return prepared


    # Remove prepared entries for specific images.
    #
    # Default behavior is intentionally image-wide: if an image was preloaded
    # with several different breakup modes/settings, clearing that image removes
    # all of its variants. This makes scene cleanup easy and avoids leaving one
    # forgotten configuration sitting in memory.
    #
    # You can also pass mode/separation/radius together when you want to remove
    # exactly one cache entry and leave the other variants intact.
    def breakup_clear(
        *images,
        mode=None,
        separation=None,
        radius=None,
    ):
        if not images:
            raise ValueError(
                "breakup_clear(): pass at least one image, or use breakup_clear_all()."
            )

        exact = (
            mode is not None
            and separation is not None
            and radius is not None
        )

        if any(value is not None for value in (mode, separation, radius)) and not exact:
            raise ValueError(
                "breakup_clear(): mode, separation, and radius must either all be "
                "provided or all be omitted."
            )

        removed = 0

        for image in images:
            if exact:
                key = _breakup_preload_key(
                    image,
                    mode,
                    separation,
                    radius,
                )
                if _BREAKUP_PRELOAD_CACHE.pop(key, None) is not None:
                    removed += 1
                    _breakup_debug(
                        "CACHE CLEAR exact image=%r key=%r cache_size_now=%s"
                        % (image, key, len(_BREAKUP_PRELOAD_CACHE))
                    )
                else:
                    _breakup_debug(
                        "CACHE CLEAR exact MISS image=%r key=%r cache_size_now=%s"
                        % (image, key, len(_BREAKUP_PRELOAD_CACHE))
                    )
                continue

            image_key = (
                ("name", image)
                if isinstance(image, str)
                else ("object", id(image))
            )

            keys_to_remove = [
                key
                for key in _BREAKUP_PRELOAD_CACHE.keys()
                if key[0] == image_key
            ]

            for key in keys_to_remove:
                del _BREAKUP_PRELOAD_CACHE[key]
                removed += 1

            _breakup_debug(
                "CACHE CLEAR image=%r removed=%s cache_size_now=%s"
                % (image, len(keys_to_remove), len(_BREAKUP_PRELOAD_CACHE))
            )

        return removed


    # Clear every persistent preload entry. This is the 'empty the whole shelf'
    # operation and is useful between chapters/scenes when many cached effects
    # are no longer needed.
    def breakup_clear_all():
        count = len(_BREAKUP_PRELOAD_CACHE)

        _BREAKUP_PRELOAD_CACHE.clear()

        _breakup_debug(
            "CACHE CLEAR ALL removed=%s cache_size_now=%s"
            % (count, len(_BREAKUP_PRELOAD_CACHE))
        )

        return count


    # =====================================================================
    # LIVE DISPLAYABLE
    # =====================================================================
    #
    # Once the public breakup() function creates this object, Ren'Py calls
    # render() repeatedly as the animation advances.
    # The important difference from a traditional implementation is that
    # render() reuses the already-prepared texture/grid/mesh data.
    # =====================================================================
    # -------------------------------------------------------------------------
    # MAIN DISPLAYABLE
    # -------------------------------------------------------------------------

    # This object is the thing Ren'Py actually puts on screen.
    class BreakupDisplayable(renpy.Displayable):

        def __init__(
            self,
            widget,
            delay,
            reverse,
            mode,
            separation,
            radius,
            **properties
        ):
            # Keep a stopwatch around the setup so debug logs can show whether
            # object construction itself ever becomes unexpectedly expensive.
            import time as _breakup_time
            t0 = _breakup_time.perf_counter()

            # `_breakup_preparing` is only used by the internal preload helper.
            # `_breakup_preloaded` is the finished package handed over by breakup().
            preparing = bool(properties.pop("_breakup_preparing", False))
            prepared = properties.pop("_breakup_preloaded", None)

            _breakup_debug(
                "DISPLAYABLE INIT start image=%r id=%s preparing=%s prepared_passed=%s"
                % (widget, id(widget), preparing, prepared is not None)
            )

            # A normal live animation is not allowed to start from scratch.
            # If the preload package is missing, fail immediately instead of
            # hiding a large first-frame stall behind lazy initialization.
            if prepared is None and not preparing:
                _breakup_debug(
                    "DISPLAYABLE INIT ERROR: no preload payload for image=%r id=%s"
                    % (widget, id(widget))
                )
                raise RuntimeError(
                    "BreakupDisplayable cannot be created without preloaded data. "
                    "Use breakup_preload() before breakup()."
                )

            super(BreakupDisplayable, self).__init__(**properties)

            # Save the user-facing animation settings on the live object.
            self.delay = float(delay)
            self.reverse = bool(reverse)
            self.mode = mode
            self.separation = int(separation)
            self.radius = float(radius)
            self._breakup_preloading = preparing
            self._breakup_finished = False

            # Runtime state. The grid/texture are initially empty because the
            # prepared package is applied just below.
            self.grid = None
            self._preloaded_source_texture = None
            self.width = 0
            self.height = 0

            self._cache_cells_x = None
            self._cache_cells_y = None
            self._cache_mode = None
            self._cache_w = None
            self._cache_h = None

            # Unpack the preload package and attach references to this live
            # Displayable. The persistent cache keeps its own references, so this
            # animation is borrowing the prepared resources rather than consuming
            # them.
            if prepared is not None:
                (
                    prepared_widget,
                    prepared_width,
                    prepared_height,
                    prepared_grid,
                    prepared_source_texture,
                ) = prepared

                self.widget = prepared_widget
                self.width = prepared_width
                self.height = prepared_height
                self.grid = prepared_grid
                self._preloaded_source_texture = prepared_source_texture

                self._cache_cells_x = prepared_grid.cells_x
                self._cache_cells_y = prepared_grid.cells_y
                self._cache_mode = mode
                self._cache_w = prepared_width
                self._cache_h = prepared_height

                _breakup_debug(
                    "DISPLAYABLE INIT APPLIED preload image=%r widget_id=%s texture_id=%s grid_id=%s meshes=%s size=%sx%s"
                    % (
                        widget,
                        id(self.widget),
                        id(self._preloaded_source_texture),
                        id(self.grid),
                        len(self.grid.mesh_groups),
                        self.width,
                        self.height,
                    )
                )
                # This branch is only for the internal preload builder. A normal
                # user-facing breakup() should always arrive with `prepared` data.
            else:
                # Only the internal preload builder is allowed to construct a
                # displayable without a user-facing preload payload. It still
                # must not be rendered as a normal animation.
                self.widget = widget

            _breakup_debug(
                "DISPLAYABLE INIT done dt=%.6fs image=%r id=%s"
                % (_breakup_time.perf_counter() - t0, widget, id(widget))
            )


            # Ren'Py can serialize displayables. GPU objects such as a texture or
            # live mesh should not be serialized as Python state, so strip them.
        def __getstate__(self):
            state = self.__dict__.copy()
            state["grid"] = None
            state["_preloaded_source_texture"] = None
            return state


        def __setstate__(self, state):
            self.__dict__.update(state)


            # Build the rectangular grid that covers the source image.
            # `separation` is basically the tile size: smaller values mean more
            # cells and therefore more geometry.
        def _ensure_grid(self, w, h):
            cell_size = self.separation

            cells_x = int((w + cell_size - 1) // cell_size) + 1
            cells_y = int((h + cell_size - 1) // cell_size) + 1
            total = cells_x * cells_y

            g = BreakupGrid()

            g.cells = [BreakupCell() for _ in range(total)]
            g.cell_size = cell_size
            g.cells_x = cells_x
            g.cells_y = cells_y
            # Diagonals are used as the effect's sweep order. Each diagonal groups
            # cells that start breaking around the same part of the animation.
            g.diag_count = cells_x + cells_y - 1

            g.diag_slices = [None] * g.diag_count
            g.diag_min_state = [1e30] * g.diag_count

            g.mode = None
            g.total_frames = float(
                BREAKUP_DISSOLVE_FRAMES + BREAKUP_WIPE_FRAMES
            )

            # Store inverse dimensions because the shader frequently converts
            # pixel distances to normalized texture-coordinate distances.
            g.inv_iw = 1.0 / float(w) if w > 0 else 0.0
            g.inv_ih = 1.0 / float(h) if h > 0 else 0.0
            g.radius = self.radius

            # The radius controls how round the breakup fragments can become.
            # More radius -> potentially more circle segments -> more geometry.
            if self.radius > 0.0:
                g.max_segments = int(
                    _CIRC_CONST * _sqrt(self.radius)
                ) + 1
                if g.max_segments < 2:
                    g.max_segments = 2
                elif g.max_segments > 99:
                    g.max_segments = 99
            else:
                g.max_segments = 2

            self.grid = g

            self._cache_cells_x = cells_x
            self._cache_cells_y = cells_y
            self._cache_mode = None
            self._cache_w = w
            self._cache_h = h


            # Fill every cell with its permanent animation metadata.
            # This is deliberately guarded so we do this work only when the grid
            # is new or one of the settings changed.
        def _setup_cells_once(self):
            g = self.grid
            if g is None:
                return

            # Fast path: if all cached dimensions/settings still match and meshes
            # already exist, there is nothing to rebuild.
            if (
                g.mode == self.mode
                and self._cache_cells_x == g.cells_x
                and self._cache_cells_y == g.cells_y
                and self._cache_mode == self.mode
                and self._cache_w == self.width
                and self._cache_h == self.height
                and g.mesh_groups
            ):
                return

            # From here on we are calculating the information that will live in
            # the static mesh: location, UVs, timing, and movement direction.
            g.mode = self.mode
            g.total_frames = float(
                BREAKUP_DISSOLVE_FRAMES + BREAKUP_WIPE_FRAMES
            )

            diag_count = g.diag_count
            cells_x = g.cells_x
            cells_y = g.cells_y
            cells = g.cells

            diag_slices = g.diag_slices
            diag_min_state = g.diag_min_state

            for i in range(diag_count):
                diag_min_state[i] = 1e30

            f = g.cell_size
            travel = float(BREAKUP_TRAVEL_PX)

            bd_frames = float(BREAKUP_DISSOLVE_FRAMES)
            bw_frames = float(BREAKUP_WIPE_FRAMES)

            # Decode the direction flags once instead of checking them for every
            # later operation.
            mode_left = self.mode & BreakupMode.LEFT
            mode_lower = self.mode & BreakupMode.LOWER
            mode_jumble = self.mode & BreakupMode.JUMBLE

            cos_list = _COS_LIST
            sin_list = _SIN_LIST

            # Base movement direction. LEFT/LOWER flip the corresponding axis;
            # JUMBLE flips both again to vary the breakup direction.
            x_dir = -1 if mode_left else 1
            y_dir = -1 if mode_lower else 1

            if mode_jumble:
                x_dir = -x_dir
                y_dir = -y_dir

            cells_y_minus_1 = cells_y - 1
            cells_x_minus_1 = cells_x - 1
            diag_count_divisor = (
                float(diag_count - 1) if diag_count > 1 else 1.0
            )

            inv_iw = g.inv_iw
            inv_ih = g.inv_ih

            n = 0

            # Walk through the grid diagonally. This is what makes the effect look
            # like a wave/sweep instead of every tile starting at once.
            for diag in range(diag_count):
                start_idx = n

                ax_sub = diag - cells_y_minus_1
                ay_sub = diag - cells_x_minus_1

                for x in range(diag, -1, -1):
                    y = diag - x

                    if y >= cells_y or x >= cells_x:
                        continue

                    # Each cell starts with the dissolve portion of its timing.
                    cell = cells[n]
                    state = bd_frames

                    # Add a small random diagonal offset so the sweep is imperfect
                    # and organic rather than a perfectly straight marching line.
                    if diag_count > 1:
                        fake_diag = diag - int(_random() * 6.0)
                        if fake_diag < 0:
                            fake_diag = 0
                        state += (
                            fake_diag * bw_frames
                        ) / diag_count_divisor

                    # Convert logical x/y into the actual effect direction.
                    # This is where LEFT and LOWER change which side the cell appears
                    # to come from.
                    cell.grid_x = x if mode_left else (
                        cells_x - x - 1
                    )
                    cell.grid_y = (
                        cells_y - y - 1
                        if mode_lower
                        else y
                    )

                    # Pixel-space position of the cell's corner.
                    cell.base_x = float(cell.grid_x * f)
                    cell.base_y = float(cell.grid_y * f)

                    # Normalized UV position: pixel coordinates divided by image size.
                    cell.uv_s = cell.base_x * inv_iw
                    cell.uv_t = cell.base_y * inv_ih

                    # Save the diagonal/timing information so the shader knows when
                    # this specific cell becomes active.
                    cell.diagonal = diag
                    cell.state = state

                    if state < diag_min_state[diag]:
                        diag_min_state[diag] = state

                    # Calculate a rough vector from the diagonal toward this cell.
                    # Its angle becomes the basis for the fragment's travel direction.
                    ax = x if ax_sub <= 0 else x - ax_sub
                    ay = y if ay_sub <= 0 else y - ay_sub

                    angle = (
                        _atan2(-ay, ax)
                        if ax
                        else 1.5707963267948966
                    )

                    # Add a small random angle wobble so neighboring pieces do not
                    # all fly out in exactly the same direction.
                    angle += _ANGLE_STEP * (
                        int(_random() * 51.0) - 25
                    )

                    deg = int(angle * _INV_180_PI) & 359

                    # Store direction once in normalized form, then also store the
                    # actual pixel travel distance. The GPU can interpolate that
                    # movement later without rebuilding the mesh.
                    cell.move_x = x_dir * cos_list[deg]
                    cell.move_y = y_dir * sin_list[deg]
                    cell.move_px_x = cell.move_x * travel
                    cell.move_px_y = cell.move_y * travel

                    n += 1

                diag_slices[diag] = (start_idx, n)

            self._cache_cells_x = cells_x
            self._cache_cells_y = cells_y
            self._cache_mode = self.mode
            self._cache_w = self.width
            self._cache_h = self.height

            # Finally turn the prepared cell data into the static Mesh2 groups.
            _breakup_build_static_meshes(g)


            # Given the current animation frame, determine how far through the
            # diagonal sweep we have progressed.
        def _max_active_diagonal(self, frame):
            g = self.grid

            threshold = float(
                max(BREAKUP_DISSOLVE_FRAMES, BREAKUP_MOVE_FRAMES)
            )

            diag_min_state = g.diag_min_state

            max_active = -1

            for d in range(g.diag_count):
                if (diag_min_state[d] - frame) < threshold:
                    max_active = d

            return 0 if max_active < 0 else max_active


            # Convert a simple 0..1 progress factor into the exact frame number
            # expected by the shader, then package the small set of runtime values
            # the GPU needs.
        def _build_meshes(self, factor):
            self._setup_cells_once()

            g = self.grid
            frame = g.total_frames * float(factor)
            max_diag = self._max_active_diagonal(frame)

            remaining_cutoff = (
                0.0001 *
                float(BREAKUP_DISSOLVE_FRAMES) /
                self.radius
                if self.radius != 0.0
                else 0.0
            )

            params = (
                float(frame),
                float(BREAKUP_DISSOLVE_FRAMES),
                float(BREAKUP_MOVE_FRAMES),
                float(remaining_cutoff),
            )

            inv_size = (g.inv_iw, g.inv_ih)
            max_diag_f = float(max_diag)

            return g.mesh_groups, params, inv_size, max_diag_f


            # Ren'Py calls this repeatedly while the effect is visible.
            # The expensive preparation should already be finished before we get here.
        def render(self, width, height, st, at):
            import time as _breakup_time
            render_t0 = _breakup_time.perf_counter()

            # If this ever fires, the internal preload helper was accidentally
            # sent through the normal render path. That would defeat the design.
            # Internal construction during breakup_preload() is never a live
            # animation. It is rendered only by the preload's explicit warmup.
            if self._breakup_preloading:
                raise RuntimeError(
                    "Internal BreakupDisplayable used by breakup_preload() was rendered unexpectedly."
                )

            # Safety net: a live effect without its prepared texture/grid is a bug.
            if self.grid is None or self._preloaded_source_texture is None:
                if self._breakup_finished:
                    _breakup_debug(
                        "RENDER finished displayable image=%r st=%.6f -> returning completion frame"
                        % (self.widget, st)
                    )
                    return renpy.Render(int(self.width), int(self.height))

                _breakup_debug(
                    "RENDER ERROR missing preload data image=%r id=%s grid=%s texture=%s"
                    % (
                        self.widget,
                        id(self.widget),
                        id(self.grid),
                        id(self._preloaded_source_texture),
                    )
                )
                raise RuntimeError(
                    "breakup render attempted without preloaded data for %r."
                    % (self.widget,)
                )

            c_w = int(self.width)
            c_h = int(self.height)

            # Once the requested delay has elapsed, the animation is over.
            # Reverse mode keeps the original texture so the image can remain visible.
            # Forward mode releases ONLY this displayable's references. The
            # persistent preload cache still owns its copy, so the resource remains
            # available for the next breakup() call until explicitly cleared.
            if self.delay <= 0.0 or st > self.delay:
                self._breakup_finished = True

                if self.reverse:
                    source_texture = self._preloaded_source_texture
                    _breakup_debug(
                        "RENDER COMPLETE reverse=True -> keeping source texture id=%s"
                        % id(source_texture)
                    )
                    result = renpy.Render(c_w, c_h)
                    result.blit(source_texture, (0, 0))
                    _breakup_debug(
                        "RENDER COMPLETE done dt=%.6fs"
                        % (_breakup_time.perf_counter() - render_t0)
                    )
                    return result

                self._preloaded_source_texture = None
                self.grid = None
                _breakup_debug(
                    "RENDER COMPLETE forward=True -> released live references; persistent cache still retains preload"
                )
                result = renpy.Render(c_w, c_h)
                _breakup_debug(
                    "RENDER COMPLETE done dt=%.6fs"
                    % (_breakup_time.perf_counter() - render_t0)
                )
                return result

            # Ren'Py gives us elapsed show-time (`st`). Turn that into 0..1 progress.
            progress = st / self.delay

            if progress < 0.0:
                progress = 0.0
            elif progress > 1.0:
                progress = 1.0

            # Smoothstep easing makes the overall breakup progress feel less mechanical.
            progress = progress * progress * (
                3.0 - 2.0 * progress
            )

            # Reverse playback simply runs the same effect from the opposite end.
            factor = (
                1.0 - progress
                if self.reverse
                else progress
            )

            # This should now be cheap compared with the old design: it mostly
            # calculates a few numbers instead of rebuilding vertices/triangles.
            mesh_t0 = _breakup_time.perf_counter()
            mesh_groups, params, inv_size, max_diag = (
                self._build_meshes(factor)
            )
            mesh_dt = _breakup_time.perf_counter() - mesh_t0

            # Build one final render surface, then layer each prepared mesh on top.
            total = renpy.Render(c_w, c_h)

            draw_t0 = _breakup_time.perf_counter()
                # Each Mesh2 becomes its own sub-render. The shader gets the current
                # time/radius settings, and the preloaded texture is sampled through it.
            for mesh_index, mesh in enumerate(mesh_groups):
                sub = renpy.Render(c_w, c_h)
                sub.mesh = mesh
                sub.add_shader("renpy.texture")
                sub.add_shader("breakup.gpu")
                sub.add_uniform("u_breakup_params", params)
                sub.add_uniform("u_breakup_inv_size", inv_size)
                sub.add_uniform("u_breakup_max_diag", max_diag)
                sub.add_uniform("u_breakup_radius", float(self.radius))
                sub.add_uniform(
                    "u_breakup_max_segments",
                    float(self.grid.max_segments),
                )
                sub.blit(self._preloaded_source_texture, (0, 0))
                total.blit(sub, (0, 0))

            # Ask Ren'Py to render this Displayable again immediately so the next
            # animation frame can use the new `st` value and recompute GPU parameters.
            draw_dt = _breakup_time.perf_counter() - draw_t0
            renpy.redraw(self, 0)

            return total



    # =====================================================================
    # PUBLIC API
    # =====================================================================
    #
    # This is the simple entry point used by the story script.
    # The intended call sequence is:
    #
    #     breakup_preload("my_image")   # do the expensive setup first
    #     show expression breakup("my_image", 1.0)
    #                                  # borrow the prepared package
    #                                  # and play the animation
    #
    # The visible call should therefore be mostly a hand-off, not a build step.
    # The same prepared package can be borrowed again later until explicitly
    # removed with breakup_clear(...) or breakup_clear_all().
    # =====================================================================
    # -------------------------------------------------------------------------
    # PUBLIC API + PRELOAD
    # -------------------------------------------------------------------------

    def breakup(
        image,
        delay=1.0,
        reverse=False,
        mode=BreakupMode.LEFT,
        separation=16,
        radius=12.0
    ):
        # First fetch the exact prepared package. If it does not exist,
        # `_breakup_take_preload()` raises instead of doing lazy work here.
        # The package remains in the persistent cache after this call.
        _breakup_debug(
            "API breakup() image=%r id=%s delay=%s reverse=%s mode=%s separation=%s radius=%s predicting=%s"
            % (
                image, id(image), delay, reverse, int(mode), separation, radius,
                bool(getattr(renpy.display.predict, "predicting", False)),
            )
        )

        prepared = _breakup_take_preload(
            image, mode, separation, radius
        )

        # Construct the live Displayable from the already-prepared package.
        # No grid/mesh regeneration is supposed to happen here.
        result = BreakupDisplayable(
            image,
            delay,
            reverse,
            mode,
            separation,
            radius,
            _breakup_preloaded=prepared,
        )

        _breakup_debug(
            "API breakup() RETURN displayable_id=%s image=%r preload_required=True cache_persistent=True"
            % (id(result), image)
        )
        return result


define e = Character("Silvie")

image bgSchoolYardDay = "images/bgSchoolYardDay.png"
image silviesprite = "images/silviesprite.png"

transform m:
    subpixel True
    zoom 1
    xalign 0.5
    yalign 0.0
    alpha 1.0

# The game starts here.
label start:

    window show
    "Click to preload breakup images"
    window hide

    show text "Preloading background and sprite..."
    $ breakup_preload("bgSchoolYardDay", "silviesprite", mode=0)
    hide text

    window show

    "Click to start."

    window hide

    show expression breakup("bgSchoolYardDay", 7, True, 0) as exampleBG

    pause 8

    window show

    "Click to continue."

    window hide

    show expression breakup("silviesprite", 7, True, 0) as example at m

    pause 7

    window show

    # These display lines of dialogue.

    e "You've just witnessed Breakup. Click to clear the scene."

    show expression breakup("silviesprite", 7, False, 0) as example at m

    pause 7

    show expression breakup("bgSchoolYardDay", 7, False, 0) as exampleBG

    pause 7

    $ breakup_clear_all()

    show text "memory cleared"
    pause 2
    hide text

    # This ends the game.

    return

