NAME          2m_4j_rep1
ROWS
 N  OBJ
 L  BUDGET
 L  ASSIGN_UP_1
 G  ASSIGN_LO_1
 L  ASSIGN_UP_2
 G  ASSIGN_LO_2
 L  ASSIGN_UP_3
 G  ASSIGN_LO_3
 L  ASSIGN_UP_4
 G  ASSIGN_LO_4
 G  MAKESPAN_1
 G  MAKESPAN_2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x_1        BUDGET     58
    x_1        ASSIGN_UP_1   -1.0
    x_1        ASSIGN_LO_1   -1.0
    x_2        BUDGET     19
    x_2        ASSIGN_UP_2   -1.0
    x_2        ASSIGN_LO_2   -1.0
    x_3        BUDGET     39
    x_3        ASSIGN_UP_3   -1.0
    x_3        ASSIGN_LO_3   -1.0
    x_4        BUDGET     24
    x_4        ASSIGN_UP_4   -1.0
    x_4        ASSIGN_LO_4   -1.0
    y_hat      OBJ        1.0
    y_1_1      ASSIGN_UP_1   1.0
    y_1_1      ASSIGN_LO_1   1.0
    y_1_1      MAKESPAN_1   12
    y_1_2      ASSIGN_UP_1   1.0
    y_1_2      ASSIGN_LO_1   1.0
    y_1_2      MAKESPAN_2   12
    y_2_1      ASSIGN_UP_2   1.0
    y_2_1      ASSIGN_LO_2   1.0
    y_2_1      MAKESPAN_1   33
    y_2_2      ASSIGN_UP_2   1.0
    y_2_2      ASSIGN_LO_2   1.0
    y_2_2      MAKESPAN_2   33
    y_3_1      ASSIGN_UP_3   1.0
    y_3_1      ASSIGN_LO_3   1.0
    y_3_1      MAKESPAN_1   34
    y_3_2      ASSIGN_UP_3   1.0
    y_3_2      ASSIGN_LO_3   1.0
    y_3_2      MAKESPAN_2   34
    y_4_1      ASSIGN_UP_4   1.0
    y_4_1      ASSIGN_LO_4   1.0
    y_4_1      MAKESPAN_1   9
    y_4_2      ASSIGN_UP_4   1.0
    y_4_2      ASSIGN_LO_4   1.0
    y_4_2      MAKESPAN_2   9
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS        BUDGET     91.0
    RHS        ASSIGN_UP_1   0.0
    RHS        ASSIGN_LO_1   0.0
    RHS        ASSIGN_UP_2   0.0
    RHS        ASSIGN_LO_2   0.0
    RHS        ASSIGN_UP_3   0.0
    RHS        ASSIGN_LO_3   0.0
    RHS        ASSIGN_UP_4   0.0
    RHS        ASSIGN_LO_4   0.0
    RHS        MAKESPAN_1   0.0
    RHS        MAKESPAN_2   0.0
BOUNDS
 LI BOUND      x_1       0
 LI BOUND      x_2       0
 LI BOUND      x_3       0
 LI BOUND      x_4       0
 LO BOUND      y_hat      0
 LI BOUND      y_1_1     0
 LI BOUND      y_1_2     0
 LI BOUND      y_2_1     0
 LI BOUND      y_2_2     0
 LI BOUND      y_3_1     0
 LI BOUND      y_3_2     0
 LI BOUND      y_4_1     0
 LI BOUND      y_4_2     0
ENDATA
