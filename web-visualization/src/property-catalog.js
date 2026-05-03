const PROPERTY_INFO = {
  axis_reflection_symmetric: {
    label: "Axis Reflection Symmetric",
    category: "Reflection",
    description: "Invariant under both left-right and top-bottom reflection.",
  },
  black_white_symmetric: {
    label: "Black/White Symmetric",
    category: "State Complement",
    description: "Complementing all nine inputs complements the output.",
  },
  antitone: {
    label: "Antitone",
    category: "Order",
    description: "Flipping any neighborhood bit from 0 to 1 cannot increase the output.",
  },
  a_permutive: {
    label: "A Permutive",
    category: "Permutivity",
    description: "Toggling the south-west corner a flips the output.",
  },
  b_permutive: {
    label: "B Permutive",
    category: "Permutivity",
    description: "Toggling the south neighbor b flips the output.",
  },
  c_permutive: {
    label: "C Permutive",
    category: "Permutivity",
    description: "Toggling the south-east corner c flips the output.",
  },
  center_antitone: {
    label: "Center Antitone",
    category: "Order",
    description: "Flipping the center cell from 0 to 1 cannot increase the output.",
  },
  center_blind: {
    label: "Center Blind",
    category: "Neighborhood Restriction",
    description: "The local rule ignores the current state of the center cell.",
  },
  center_monotone: {
    label: "Center Monotone",
    category: "Order",
    description: "Flipping the center cell from 0 to 1 cannot decrease the output.",
  },
  center_permutive: {
    label: "Center Permutive",
    category: "Permutivity",
    description: "For every fixed outer neighborhood, toggling the center cell flips the output.",
  },
  diagonal_monotone: {
    label: "Diagonal Monotone",
    category: "Order",
    description: "Monotone increasing in the four diagonal neighbors.",
  },
  diagonal_only: {
    label: "Diagonal Only",
    category: "Neighborhood Restriction",
    description: "The orthogonal neighbors are irrelevant; only center and diagonals matter.",
  },
  diagonal_von_neumann: {
    label: "Diagonal Von Neumann Embedded",
    category: "Neighborhood Restriction",
    description: "Alias of Diagonal Only: only the center and four diagonal sites matter.",
  },
  diagonal_reflection_symmetric: {
    label: "Diagonal Reflection Symmetric",
    category: "Reflection",
    description: "Invariant under both diagonal mirror symmetries.",
  },
  diagonal_totalistic: {
    label: "Diagonal Totalistic",
    category: "Count Based",
    description: "Depends only on the center cell and the number of live diagonal neighbors.",
  },
  isotropic_non_totalistic: {
    label: "Isotropic Non-Totalistic",
    category: "Symmetry",
    description: "Full dihedral D4 symmetry of the Moore neighborhood.",
  },
  life_like: {
    label: "Life-Like",
    category: "Count Based",
    description: "Conventional outer-totalistic family for binary Moore-neighborhood rules.",
  },
  monotone: {
    label: "Monotone",
    category: "Order",
    description: "Flipping any neighborhood bit from 0 to 1 cannot decrease the output.",
  },
  ortho_diag_semitotalistic: {
    label: "Ortho/Diag Semi-Totalistic",
    category: "Count Based",
    description: "Depends on the center plus the multisets of orthogonal and diagonal neighbors.",
  },
  orthogonal_totalistic: {
    label: "Orthogonal Totalistic",
    category: "Count Based",
    description: "Depends only on the center cell and the number of live orthogonal neighbors.",
  },
  orthogonal_monotone: {
    label: "Orthogonal Monotone",
    category: "Order",
    description: "Monotone increasing in the four orthogonal neighbors.",
  },
  outer_monotone: {
    label: "Outer Monotone",
    category: "Order",
    description: "Monotone increasing in all eight outer neighbors.",
  },
  outer_totalistic: {
    label: "Outer Totalistic",
    category: "Count Based",
    description: "Depends only on the center cell and the total number of live outer neighbors.",
  },
  reflection_anti_diagonal: {
    label: "Anti-Diagonal Reflection",
    category: "Reflection",
    description: "Mirror symmetry across the top-right to bottom-left diagonal.",
  },
  reflection_left_right: {
    label: "Left-Right Reflection",
    category: "Reflection",
    description: "Mirror symmetry across the vertical axis of the neighborhood.",
  },
  reflection_main_diagonal: {
    label: "Main-Diagonal Reflection",
    category: "Reflection",
    description: "Mirror symmetry across the top-left to bottom-right diagonal.",
  },
  reflection_top_bottom: {
    label: "Top-Bottom Reflection",
    category: "Reflection",
    description: "Mirror symmetry across the horizontal axis of the neighborhood.",
  },
  rotation_180: {
    label: "180° Rotation Symmetric",
    category: "Rotation",
    description: "Invariant under a half-turn of the Moore neighborhood.",
  },
  rotation_symmetric: {
    label: "90° Rotation Symmetric",
    category: "Rotation",
    description: "Invariant under quarter turns; this also implies 180° and 270° symmetry.",
  },
  totalistic: {
    label: "Totalistic",
    category: "Count Based",
    description: "Depends only on the total number of live cells in the entire 3×3 neighborhood.",
  },
  von_neumann: {
    label: "Von Neumann Embedded",
    category: "Neighborhood Restriction",
    description: "The four corners are irrelevant; only the five-cell cross neighborhood matters.",
  },
  t_permutive: {
    label: "T Permutive",
    category: "Permutivity",
    description: "Toggling the west neighbor t flips the output.",
  },
  w_permutive: {
    label: "W Permutive",
    category: "Permutivity",
    description: "Toggling the east neighbor w flips the output.",
  },
  x_permutive: {
    label: "X Permutive",
    category: "Permutivity",
    description: "Toggling the north-west corner x flips the output.",
  },
  y_permutive: {
    label: "Y Permutive",
    category: "Permutivity",
    description: "Toggling the north neighbor y flips the output.",
  },
  z_permutive: {
    label: "Z Permutive",
    category: "Permutivity",
    description: "Toggling the north-east corner z flips the output.",
  },
};

export function getPropertyInfo(name) {
  return (
    PROPERTY_INFO[name] ?? {
      label: name.replaceAll("_", " "),
      category: "Other",
      description: "Additional property imported from the solver metadata.",
    }
  );
}

export function enrichPropertyDefinition(property, count = 0) {
  const info = getPropertyInfo(property.name);
  return {
    ...property,
    ...info,
    count,
    bitMask: 2 ** property.bit,
    machineName: property.name,
  };
}
