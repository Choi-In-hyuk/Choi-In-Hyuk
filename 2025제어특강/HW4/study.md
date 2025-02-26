 interior-point-convex (기본 설정)

options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex');

    초기값(u0)을 무시하고 자체적으로 최적화 시작.
    대규모 문제에서 빠르게 수렴.

(2) active-set (초기값을 사용)

options = optimoptions('quadprog', 'Algorithm', 'active-set');

    초기값(u0)을 고려하여 최적화를 시작.
    초기값이 최적해와 가까우면 더 빠를 수 있음.
    하지만, 지역 최적해에 빠질 가능성이 있음.
