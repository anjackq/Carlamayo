from module.route_authorization import (
    CandidateLaneFact,
    RouteStatus,
    assess_route_candidate,
    combine_execution_constraints,
)
from module.route_navigation import RoutePoint, build_route_plan


def _route():
    return build_route_plan(
        [
            RoutePoint((0, 0, 0), 1, 0, -1, False, "LANEFOLLOW"),
            RoutePoint((1, 0, 0), 1, 0, -1, False, "LANEFOLLOW"),
            RoutePoint((2, 0, 0), 2, 0, -1, True, "RIGHT"),
            RoutePoint((3, 0, 0), 3, 0, -1, True, "RIGHT"),
            RoutePoint((4, 0, 0), 3, 0, -1, False, "LANEFOLLOW"),
        ]
    )


def _fact(road, lane=-1, junction=False):
    return CandidateLaneFact(road, 0, lane, junction)


def test_natural_route_curvature_remains_authorized():
    assessment = assess_route_candidate(
        route=_route(),
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[
            (0.5, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ],
        waypoint_times_s=[0.5, 1.0, 1.5, 2.0],
        source_simulation_time_s=0.0,
        lane_facts=[_fact(1), _fact(1), _fact(2, junction=True), _fact(3, junction=True)],
    )
    assert assessment.near_term_route_status is RouteStatus.MATCH
    assert assessment.full_path_route_status is RouteStatus.MATCH
    assert assessment.last_authorized_waypoint_index == 3


def test_exact_lane_identity_bridges_grp_transition_overlap():
    route = build_route_plan(
        [
            RoutePoint((0, 0, 0), 28, 0, 3, False, "LANEFOLLOW"),
            RoutePoint((2, 0, 0), 1691, 0, 3, True, "LANEFOLLOW"),
            RoutePoint((3, 0, 0), 1691, 0, 3, True, "LANEFOLLOW"),
        ]
    )
    assessment = assess_route_candidate(
        route=route,
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[
            (1.1, 0, 0),
            (1.9, 0, 0),
            (2.1, 0, 0),
            (3.0, 0, 0),
        ],
        waypoint_times_s=[0.5, 1.0, 1.5, 2.0],
        source_simulation_time_s=0.0,
        lane_facts=[
            _fact(28, lane=3),
            _fact(28, lane=3),
            _fact(1691, lane=3, junction=True),
            _fact(1691, lane=3, junction=True),
        ],
    )
    assert assessment.near_term_route_status is RouteStatus.MATCH
    assert assessment.full_path_route_status is RouteStatus.MATCH
    assert assessment.last_authorized_waypoint_index == 3


def test_bounded_future_connector_overlap_is_canonicalized():
    route = build_route_plan(
        [
            RoutePoint((0, 0, 0), 1577, 0, -3, True, "RIGHT"),
            RoutePoint((1, 0, 0), 1577, 0, -3, True, "RIGHT"),
            RoutePoint((10, 0, 0), 1608, 1, -1, True, "RIGHT"),
            RoutePoint((20, 0, 0), 61, 0, -1, False, "LANEFOLLOW"),
        ]
    )
    assessment = assess_route_candidate(
        route=route,
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[
            (0.2, 0, 0),
            (0.8, 0, 0),
            (1.2, 0, 0),
        ],
        waypoint_times_s=[0.5, 1.0, 1.5],
        source_simulation_time_s=0.0,
        lane_facts=[
            _fact(1608, lane=-3, junction=True),
            _fact(1608, lane=-3, junction=True),
            _fact(1608, lane=-3, junction=True),
        ],
    )

    assert assessment.near_term_route_status is RouteStatus.MATCH
    assert assessment.full_path_route_status is RouteStatus.MATCH
    assert "junction_topology_overlap_canonicalized" in assessment.reason_codes


def test_future_connector_overlap_outside_route_corridor_is_deviation():
    route = build_route_plan(
        [
            RoutePoint((0, 0, 0), 1577, 0, -3, True, "RIGHT"),
            RoutePoint((1, 0, 0), 1577, 0, -3, True, "RIGHT"),
            RoutePoint((10, 0, 0), 1608, 1, -1, True, "RIGHT"),
        ]
    )
    assessment = assess_route_candidate(
        route=route,
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[(0.2, 1.5, 0)],
        waypoint_times_s=[0.5],
        source_simulation_time_s=0.0,
        lane_facts=[_fact(1608, lane=-3, junction=True)],
    )

    assert assessment.near_term_route_status is RouteStatus.DEVIATE
    assert "unauthorized_junction_branch" in assessment.reason_codes


def test_wrong_junction_branch_is_deviation_and_prefix_is_bounded():
    assessment = assess_route_candidate(
        route=_route(),
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[
            (0.5, 0, 0),
            (1, 0, 0),
            (2, 0.2, 0),
            (3, 1, 0),
        ],
        waypoint_times_s=[0.5, 1.0, 1.5, 2.0],
        source_simulation_time_s=0.0,
        lane_facts=[_fact(1), _fact(1), _fact(2, junction=True), _fact(99, junction=True)],
    )
    assert assessment.near_term_route_status is RouteStatus.MATCH
    assert assessment.full_path_route_status is RouteStatus.DEVIATE
    assert assessment.last_authorized_waypoint_index == 2
    assert assessment.first_deviation_waypoint_index == 3
    assert assessment.route_speed_cap_mps is not None
    assert "unauthorized_junction_branch" in assessment.reason_codes


def test_adjacent_driving_lane_is_unauthorized_lane_change():
    assessment = assess_route_candidate(
        route=_route(),
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[(0.5, 0, 0), (1, 0, 0)],
        waypoint_times_s=[0.5, 1.0],
        source_simulation_time_s=0.0,
        lane_facts=[_fact(1), _fact(1, lane=-2)],
    )
    assert assessment.near_term_route_status is RouteStatus.DEVIATE
    assert assessment.lane_change_detected
    assert assessment.last_authorized_waypoint_index == 0


def test_unknown_map_query_fails_closed():
    assessment = assess_route_candidate(
        route=_route(),
        current_route_index=0,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[(0.5, 0, 0)],
        waypoint_times_s=[0.5],
        source_simulation_time_s=0.0,
        lane_facts=[CandidateLaneFact(None, None, None, None)],
    )
    assert assessment.near_term_route_status is RouteStatus.UNKNOWN
    assert assessment.last_authorized_waypoint_index is None


def test_elapsed_bad_prefix_does_not_poison_current_execution_window():
    assessment = assess_route_candidate(
        route=_route(),
        current_route_index=2,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=[
            (0.0, 2.0, 0),
            (1.0, 2.0, 0),
            (2.0, 0.0, 0),
            (3.0, 0.0, 0),
        ],
        waypoint_times_s=[0.5, 1.0, 1.5, 2.0],
        source_simulation_time_s=0.0,
        current_simulation_time_s=1.0,
        lane_facts=[
            _fact(99),
            _fact(99),
            _fact(2, junction=True),
            _fact(3, junction=True),
        ],
    )

    assert assessment.near_term_route_status is RouteStatus.MATCH
    assert assessment.full_path_route_status is RouteStatus.MATCH
    assert assessment.first_deviation_waypoint_index is None


def test_controller_authority_is_strict_road_route_intersection():
    speed_cap, waypoint_index = combine_execution_constraints(
        road_speed_cap_mps=4.0,
        route_speed_cap_mps=2.5,
        road_last_authorized_index=20,
        route_last_authorized_index=12,
    )
    assert speed_cap == 2.5
    assert waypoint_index == 12
