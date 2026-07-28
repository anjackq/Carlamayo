import pytest

from module.route_navigation import (
    NavigationAction,
    RouteNavigationTracker,
    RoutePoint,
    RouteTrackerStatus,
    build_route_plan,
    format_navigation_prompt,
    quantize_prompt_distance,
)


def _point(x, *, option="LANEFOLLOW", road=1, lane=-1, junction=False):
    return RoutePoint(
        xyz=(float(x), 0.0, 0.0),
        road_id=road,
        section_id=0,
        lane_id=lane,
        is_junction=junction,
        road_option=option,
    )


def _route():
    return build_route_plan(
        [
            _point(0),
            _point(10),
            _point(20, option="STRAIGHT", junction=True),
            _point(30),
            _point(40, option="RIGHT", road=2, junction=True),
            _point(50, road=3),
        ]
    )


def test_prompt_distance_buckets_and_close_range_omission():
    assert quantize_prompt_distance(34.9) == 30
    assert quantize_prompt_distance(35.0) == 40
    assert quantize_prompt_distance(15.0) is None
    assert format_navigation_prompt(
        NavigationAction.RIGHT,
        29.0,
    ) == "Turn right at the next junction in 30m."
    assert format_navigation_prompt(
        NavigationAction.STRAIGHT,
        10.0,
    ) == "Continue straight at the next junction."


def test_route_rejects_lane_change_edges():
    with pytest.raises(ValueError, match="lane-change"):
        build_route_plan([_point(0), _point(1, option="CHANGELANELEFT")])


def test_zero_length_grp_maneuver_transition_is_preserved():
    route = build_route_plan(
        [
            _point(0),
            _point(1),
            _point(1, option="RIGHT", road=2, junction=True),
            _point(2, option="RIGHT", road=2, junction=True),
        ]
    )
    tracker = RouteNavigationTracker(route)
    update = tracker.update(
        (0.0, 0.0, 0.0),
        source_frame_id=1,
        source_simulation_time_s=0.1,
    )
    assert update.context.action is NavigationAction.RIGHT
    assert update.context.target_route_index == 2


def test_close_straight_entry_and_turn_exit_are_coalesced():
    route = build_route_plan(
        [
            _point(0),
            _point(10),
            _point(20, option="STRAIGHT", junction=True),
            _point(30, road=2),
            _point(36, option="RIGHT", road=3, junction=True),
            _point(46, road=4),
        ]
    )
    tracker = RouteNavigationTracker(route)
    update = tracker.update(
        (0.0, 0.0, 0.0),
        source_frame_id=1,
        source_simulation_time_s=0.1,
    )

    assert update.context.action is NavigationAction.RIGHT
    assert update.context.target_route_index == 4
    assert update.context.text == "Turn right at the next junction in 40m."


def test_tracker_association_is_monotonic_and_epoch_changes_by_maneuver():
    tracker = RouteNavigationTracker(_route())
    initial = tracker.update(
        (0.1, 0.0, 0.0),
        source_frame_id=10,
        source_simulation_time_s=1.0,
    )
    assert initial.context.action is NavigationAction.STRAIGHT
    assert initial.context.conditioning_epoch == 0
    assert not initial.epoch_changed

    same_maneuver = tracker.update(
        (9.9, 0.0, 0.0),
        source_frame_id=20,
        source_simulation_time_s=2.0,
    )
    assert same_maneuver.context.action is NavigationAction.STRAIGHT
    assert same_maneuver.context.conditioning_epoch == 0

    next_maneuver = tracker.update(
        (30.1, 0.0, 0.0),
        source_frame_id=30,
        source_simulation_time_s=3.0,
    )
    assert next_maneuver.context.action is NavigationAction.RIGHT
    assert next_maneuver.context.conditioning_epoch == 1
    assert next_maneuver.epoch_changed

    # A noisy pose behind the vehicle cannot move association backwards.
    tracker.update(
        (20.1, 0.0, 0.0),
        source_frame_id=31,
        source_simulation_time_s=3.1,
    )
    assert tracker.route_index >= 3


def test_route_loss_is_fail_closed_without_epoch_change():
    tracker = RouteNavigationTracker(_route())
    update = tracker.update(
        (0.0, 4.0, 0.0),
        source_frame_id=1,
        source_simulation_time_s=0.1,
    )
    assert update.context.tracker_status is RouteTrackerStatus.ROUTE_UNAVAILABLE
    assert update.context.text == ""
    assert update.context.conditioning_epoch == 0


def test_exact_lane_identity_prevents_premature_connector_transition():
    route = build_route_plan(
        [
            _point(0.0, road=28, lane=3),
            _point(2.0, road=1691, lane=3, junction=True),
            _point(3.0, road=1691, lane=3, junction=True),
            _point(10.0, road=25, lane=3),
        ]
    )
    tracker = RouteNavigationTracker(route)

    incoming = tracker.update(
        (1.1, 0.0, 0.0),
        source_frame_id=1,
        source_simulation_time_s=0.1,
        ego_lane_identity=(28, 0, 3),
        require_lane_identity=True,
    )
    assert incoming.context.tracker_status is RouteTrackerStatus.AVAILABLE
    assert tracker.route_index == 0
    assert incoming.route_distance_m == pytest.approx(1.1)

    connector = tracker.update(
        (1.9, 0.0, 0.0),
        source_frame_id=2,
        source_simulation_time_s=0.2,
        ego_lane_identity=(1691, 0, 3),
        require_lane_identity=True,
    )
    assert connector.context.tracker_status is RouteTrackerStatus.AVAILABLE
    assert tracker.route_index == 1
    assert connector.route_distance_m == pytest.approx(0.1)


def test_required_lane_identity_mismatch_is_route_unavailable():
    tracker = RouteNavigationTracker(_route())
    update = tracker.update(
        (0.1, 0.0, 0.0),
        source_frame_id=1,
        source_simulation_time_s=0.1,
        ego_lane_identity=(99, 0, -1),
        require_lane_identity=True,
    )
    assert update.context.tracker_status is RouteTrackerStatus.ROUTE_UNAVAILABLE
    assert tracker.route_index == 0


def test_arrival_context_requests_destination_stop():
    tracker = RouteNavigationTracker(_route())
    update = tracker.update(
        (50.0, 0.0, 0.0),
        source_frame_id=100,
        source_simulation_time_s=10.0,
    )
    assert update.context.action is NavigationAction.ARRIVE
    assert update.context.tracker_status is RouteTrackerStatus.ARRIVED
    assert update.context.text == "Stop at the destination."
