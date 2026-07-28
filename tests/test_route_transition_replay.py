import json
from dataclasses import replace
from pathlib import Path

from module.route_authorization import (
    CandidateLaneFact,
    RouteStatus,
    assess_route_candidate,
)
from module.route_navigation import (
    RouteNavigationTracker,
    RoutePoint,
    RouteTrackerStatus,
    build_route_plan,
)
from scripts.build_route_transition_replay_fixture import validate_fixture


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "job_22921097_route_transition.json"
)


def _load_replay():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    validate_fixture(payload)
    route = build_route_plan(
        [
            RoutePoint(
                xyz=tuple(point["xyz"]),
                road_id=point["road_id"],
                section_id=point["section_id"],
                lane_id=point["lane_id"],
                is_junction=point["is_junction"],
                road_option=point["road_option"],
            )
            for point in payload["route_points"]
        ]
    )
    lane_facts = tuple(
        CandidateLaneFact(
            fact["road_id"],
            fact["section_id"],
            fact["lane_id"],
            fact["is_junction"],
        )
        for fact in payload["candidate_lane_facts"]
    )
    return payload, route, lane_facts


def test_job_22921097_identity_aware_transition_is_not_absorbing_stop():
    replay, route, lane_facts = _load_replay()
    ego_fact = replay["ego_lane_fact"]
    ego_identity = (
        ego_fact["road_id"],
        ego_fact["section_id"],
        ego_fact["lane_id"],
    )
    tracker = RouteNavigationTracker(route)
    update = tracker.update(
        replay["ego_xyz"],
        source_frame_id=replay["source_loop_tick_id"],
        source_simulation_time_s=replay["source_simulation_time_s"],
        ego_lane_identity=ego_identity,
        require_lane_identity=True,
    )

    assert update.context.tracker_status is RouteTrackerStatus.AVAILABLE
    assert tracker.route_index == replay["expected_identity_route_index"]
    assert tracker.route_index < replay["telemetry_route_index"]

    assessment = assess_route_candidate(
        route=route,
        current_route_index=tracker.route_index,
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=replay["selected_trajectory_world"],
        waypoint_times_s=replay["waypoint_times_s"],
        source_simulation_time_s=replay["source_simulation_time_s"],
        current_simulation_time_s=replay["source_simulation_time_s"],
        lane_facts=lane_facts,
    )
    assert assessment.current_route_status.value == replay["expected"][
        "current_route_status"
    ]
    assert assessment.near_term_route_status.value == replay["expected"][
        "near_term_route_status"
    ]
    assert assessment.full_path_route_status.value == replay["expected"][
        "full_path_route_status"
    ]


def test_job_22921097_wrong_connector_remains_unauthorized():
    replay, route, lane_facts = _load_replay()
    first_connector = next(
        index
        for index, fact in enumerate(lane_facts)
        if fact.is_junction and fact.road_id != replay["ego_lane_fact"]["road_id"]
    )
    mutated = list(lane_facts)
    mutated[first_connector] = replace(
        mutated[first_connector],
        road_id=999999,
    )
    assessment = assess_route_candidate(
        route=route,
        current_route_index=replay["expected_identity_route_index"],
        current_route_status=RouteStatus.MATCH,
        trajectory_world_points=replay["selected_trajectory_world"],
        waypoint_times_s=replay["waypoint_times_s"],
        source_simulation_time_s=replay["source_simulation_time_s"],
        current_simulation_time_s=replay["source_simulation_time_s"],
        lane_facts=mutated,
    )

    assert assessment.full_path_route_status.value == replay["expected"][
        "wrong_branch_status"
    ]
    assert "unauthorized_junction_branch" in assessment.reason_codes
