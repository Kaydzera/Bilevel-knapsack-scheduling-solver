NAME          2m_4j_rep0
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
    x_1        BUDGET     5
    x_1        ASSIGN_UP_1   -1
    x_1        ASSIGN_LO_1   -1
    x_2        BUDGET     8
    x_2        ASSIGN_UP_2   -1
    x_2        ASSIGN_LO_2   -1
    x_3        BUDGET     6
    x_3        ASSIGN_UP_3   -1
    x_3        ASSIGN_LO_3   -1
    x_4        BUDGET     10
    x_4        ASSIGN_UP_4   -1
    x_4        ASSIGN_LO_4   -1
    MARK0001  'MARKER'                 'INTEND'
    y_hat      OBJ        -1
    y_hat      MAKESPAN_1   1
    y_hat      MAKESPAN_2   1
    MARK0002  'MARKER'                 'INTORG'
    y_1_1      ASSIGN_UP_1   1
    y_1_1      ASSIGN_LO_1   1
    y_1_1      MAKESPAN_1   -10
    y_1_2      ASSIGN_UP_1   1
    y_1_2      ASSIGN_LO_1   1
    y_1_2      MAKESPAN_2   -10
    y_2_1      ASSIGN_UP_2   1
    y_2_1      ASSIGN_LO_2   1
    y_2_1      MAKESPAN_1   -20
    y_2_2      ASSIGN_UP_2   1
    y_2_2      ASSIGN_LO_2   1
    y_2_2      MAKESPAN_2   -20
    y_3_1      ASSIGN_UP_3   1
    y_3_1      ASSIGN_LO_3   1
    y_3_1      MAKESPAN_1   -15
    y_3_2      ASSIGN_UP_3   1
    y_3_2      ASSIGN_LO_3   1
    y_3_2      MAKESPAN_2   -15
    y_4_1      ASSIGN_UP_4   1
    y_4_1      ASSIGN_LO_4   1
    y_4_1      MAKESPAN_1   -25
    y_4_2      ASSIGN_UP_4   1
    y_4_2      ASSIGN_LO_4   1
    y_4_2      MAKESPAN_2   -25
    MARK0003  'MARKER'                 'INTEND'
RHS
    RHS        BUDGET     20
    RHS        ASSIGN_UP_1   0
    RHS        ASSIGN_LO_1   0
    RHS        ASSIGN_UP_2   0
    RHS        ASSIGN_LO_2   0
    RHS        ASSIGN_UP_3   0
    RHS        ASSIGN_LO_3   0
    RHS        ASSIGN_UP_4   0
    RHS        ASSIGN_LO_4   0
    RHS        MAKESPAN_1   0
    RHS        MAKESPAN_2   0
BOUNDS
 LI BND        x_1       0
 LI BND        x_2       0
 LI BND        x_3       0
 LI BND        x_4       0
 LO BND        y_hat      0
 LI BND        y_1_1     0
 LI BND        y_1_2     0
 LI BND        y_2_1     0
 LI BND        y_2_2     0
 LI BND        y_3_1     0
 LI BND        y_3_2     0
 LI BND        y_4_1     0
 LI BND        y_4_2     0
ENDATA
