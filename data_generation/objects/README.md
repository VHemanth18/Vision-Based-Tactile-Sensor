# objects/

One URDF per indenter, named `<shape>.urdf` (e.g. `hexagonal_indenter.urdf`), with the STL meshes in `meshes/`.

1. Open the part in `cad/solidworks/` and export it as STL (millimetres).
2. Copy `indenter_template.urdf`, change the robot name and the two mesh file names.
3. The scale `0.001 0.001 0.001` converts millimetres to metres.

Indenter names used by the dataset: asterisk_indenter, ball_array_indenter, hash_indenter, heart_indenter,
hexagonal_indenter, ring_indenter, star_indenter, triangle_indenter.
