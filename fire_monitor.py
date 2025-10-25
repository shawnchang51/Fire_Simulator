"""
Fire Model Monitoring System
============================

Real-time monitoring and visualization of fire simulation data.
Perfect for science fair demonstrations and analysis.
"""

import json
import time
from typing import List, Dict, Any
from fire_model_float import create_fire_model, EnvironmentalParameters


class FireMonitor:
    """
    Advanced monitoring system for fire simulation tracking.
    Provides real-time data collection and analysis.
    """

    def __init__(self, model, lightweight_mode=False):
        self.model = model
        self.lightweight_mode = lightweight_mode

        if not lightweight_mode:
            self.history = {
                'steps': [],
                'fire_states': [],
                'oxygen_levels': [],
                'temperatures': [],
                'smoke_density': [],
                'fuel_levels': [],
                'statistics': [],
                'changes_per_step': []
            }
        else:
            # In lightweight mode, only track minimal statistics
            self.history = {
                'steps': [],
                'statistics': [],
                'changes_per_step': []
            }
        self.step_count = 0

    def monitor_step(self, fire_state: List[List[float]]) -> Dict[str, Any]:
        """
        Monitor one simulation step and collect all tracking data.

        Returns comprehensive monitoring data for the current step.
        """
        # Run simulation step
        changes = self.model.simulate_step(fire_state)

        # Apply changes to fire state
        for pos, new_val in changes.items():
            x = int(pos.split('y')[0][1:])
            y = int(pos.split('y')[1])
            fire_state[y][x] = new_val

        # Get comprehensive statistics
        stats = self.model.get_simulation_statistics()

        # Store in history - CONDITIONAL on lightweight mode
        self.history['steps'].append(self.step_count)
        self.history['statistics'].append(stats)
        self.history['changes_per_step'].append(len(changes))

        # Only store environmental snapshots if NOT in lightweight mode
        if not self.lightweight_mode:
            # Collect environmental data snapshots (MEMORY INTENSIVE!)
            oxygen_snapshot = [row[:] for row in self.model.oxygen_map]
            temp_snapshot = [row[:] for row in self.model.temperature_map]
            smoke_snapshot = [row[:] for row in self.model.smoke_density]
            fuel_snapshot = [row[:] for row in self.model.fuel_map]

            self.history['fire_states'].append([row[:] for row in fire_state])
            self.history['oxygen_levels'].append(oxygen_snapshot)
            self.history['temperatures'].append(temp_snapshot)
            self.history['smoke_density'].append(smoke_snapshot)
            self.history['fuel_levels'].append(fuel_snapshot)

        # Prepare detailed monitoring report
        monitoring_data = {
            'step': self.step_count,
            'changes': changes,
            'cells_changed': len(changes),
            'statistics': stats,
        }

        # Only include detailed environmental data if NOT in lightweight mode
        if not self.lightweight_mode:
            monitoring_data['environmental_snapshot'] = {
                'oxygen_map': oxygen_snapshot,
                'temperature_map': temp_snapshot,
                'smoke_density_map': smoke_snapshot,
                'fuel_map': fuel_snapshot,
                'burn_time_map': [row[:] for row in self.model.burn_time]
            }
            monitoring_data['fire_analysis'] = self._analyze_fire_state(fire_state)
            monitoring_data['environmental_analysis'] = self._analyze_environment()

        self.step_count += 1
        return monitoring_data

    def _analyze_fire_state(self, fire_state: List[List[float]]) -> Dict[str, Any]:
        """Analyze current fire state for insights"""
        total_cells = self.model.rows * self.model.cols
        burning_cells = 0
        total_intensity = 0.0
        max_intensity = 0.0
        fire_perimeter = 0

        for i in range(self.model.rows):
            for j in range(self.model.cols):
                if 0 < fire_state[i][j] <= 4:
                    burning_cells += 1
                    total_intensity += fire_state[i][j]
                    max_intensity = max(max_intensity, fire_state[i][j])

                    # Check if on fire perimeter
                    neighbors = self.model._get_neighbors(i, j)
                    for ni, nj in neighbors:
                        if fire_state[ni][nj] == 0:  # Adjacent to unburned
                            fire_perimeter += 1
                            break

        burn_percentage = (burning_cells / total_cells) * 100
        avg_intensity = total_intensity / max(burning_cells, 1)

        return {
            'burning_cells': burning_cells,
            'burn_percentage': burn_percentage,
            'average_intensity': avg_intensity,
            'max_intensity': max_intensity,
            'fire_perimeter': fire_perimeter,
            'spread_rate': fire_perimeter / max(burning_cells, 1)
        }

    def _analyze_environment(self) -> Dict[str, Any]:
        """Analyze environmental conditions"""
        total_cells = self.model.rows * self.model.cols

        # Oxygen analysis
        avg_oxygen = sum(sum(row) for row in self.model.oxygen_map) / total_cells
        min_oxygen = min(min(row) for row in self.model.oxygen_map)

        # Temperature analysis
        avg_temp = sum(sum(row) for row in self.model.temperature_map) / total_cells
        max_temp = max(max(row) for row in self.model.temperature_map)

        # Smoke analysis
        total_smoke = sum(sum(row) for row in self.model.smoke_density)
        max_smoke = max(max(row) for row in self.model.smoke_density)

        # Fuel analysis
        avg_fuel = sum(sum(row) for row in self.model.fuel_map) / total_cells
        min_fuel = min(min(row) for row in self.model.fuel_map)

        return {
            'oxygen': {
                'average': avg_oxygen,
                'minimum': min_oxygen,
                'depletion_severity': max(0, (21.0 - avg_oxygen) / 21.0 * 100)
            },
            'temperature': {
                'average': avg_temp,
                'maximum': max_temp,
                'heat_buildup': max_temp - self.model.env.temperature
            },
            'smoke': {
                'total_density': total_smoke,
                'maximum_local': max_smoke,
                'visibility_impact': min(100, total_smoke * 10)
            },
            'fuel': {
                'average_remaining': avg_fuel,
                'minimum_remaining': min_fuel,
                'consumption_rate': max(0, (1.0 - avg_fuel) * 100)
            }
        }

    def print_live_monitor(self, monitoring_data: Dict[str, Any]):
        """Print real-time monitoring data to console"""
        step = monitoring_data['step']
        changes = monitoring_data['changes']
        stats = monitoring_data['statistics']
        fire_analysis = monitoring_data['fire_analysis']
        env_analysis = monitoring_data['environmental_analysis']

        print(f"\n=== STEP {step} MONITORING ===")
        print(f"Changes: {len(changes)} cells modified")

        if changes:
            print("Cell Updates:")
            for pos, val in list(changes.items())[:5]:  # Show first 5
                print(f"  {pos}: {val:.2f}")
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more")

        print(f"\nFIRE STATUS:")
        print(f"  Burning cells: {fire_analysis['burning_cells']}")
        print(f"  Burn coverage: {fire_analysis['burn_percentage']:.1f}%")
        print(f"  Avg intensity: {fire_analysis['average_intensity']:.2f}")
        print(f"  Max intensity: {fire_analysis['max_intensity']:.2f}")
        print(f"  Fire perimeter: {fire_analysis['fire_perimeter']}")

        print(f"\nENVIRONMENT:")
        print(f"  Avg oxygen: {env_analysis['oxygen']['average']:.1f}%")
        print(f"  Min oxygen: {env_analysis['oxygen']['minimum']:.1f}%")
        print(f"  Max temp: {env_analysis['temperature']['maximum']:.1f}°C")
        print(f"  Total smoke: {env_analysis['smoke']['total_density']:.2f}")
        print(f"  Avg fuel left: {env_analysis['fuel']['average_remaining']:.2f}")

        print(f"\nSTATISTICS:")
        print(f"  Fire Safety Index: {stats['fire_safety_index']:.1f}")
        print(f"  CO concentration: {stats['co_concentration_ppm']:.1f} ppm")
        print(f"  Oxygen consumed: {stats['oxygen_consumed_percent']:.1f}%")
        print(f"  Heat generated: {stats['average_temperature_rise']:.1f}°C")

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of entire simulation"""
        if not self.history['steps']:
            return {"error": "No simulation data available"}

        total_steps = len(self.history['steps'])
        final_stats = self.history['statistics'][-1]

        # Calculate trends
        changes_trend = self.history['changes_per_step']
        peak_activity_step = changes_trend.index(max(changes_trend))

        # Safety analysis
        safety_scores = [stat['fire_safety_index'] for stat in self.history['statistics']]
        worst_safety = min(safety_scores)
        safety_decline = safety_scores[0] - safety_scores[-1]

        return {
            'simulation_summary': {
                'total_steps': total_steps,
                'peak_activity_step': peak_activity_step,
                'peak_changes': max(changes_trend),
                'total_changes': sum(changes_trend)
            },
            'safety_analysis': {
                'initial_safety': safety_scores[0],
                'final_safety': safety_scores[-1],
                'worst_safety': worst_safety,
                'safety_decline': safety_decline,
                'safety_classification': self._classify_safety(worst_safety)
            },
            'environmental_impact': {
                'max_temperature_reached': max(stat['max_temperature_celsius'] for stat in self.history['statistics']),
                'total_oxygen_consumed': final_stats['oxygen_consumed_percent'],
                'peak_co_concentration': max(stat['co_concentration_ppm'] for stat in self.history['statistics']),
                'total_smoke_produced': final_stats['total_smoke_density']
            },
            'final_statistics': final_stats
        }

    def _classify_safety(self, safety_score: float) -> str:
        """Classify safety level based on score"""
        if safety_score >= 80:
            return "SAFE"
        elif safety_score >= 60:
            return "CAUTION"
        elif safety_score >= 40:
            return "DANGEROUS"
        else:
            return "CRITICAL"

    def save_monitoring_data(self, filename: str = "fire_monitoring_data.json", silent: bool = False):
        """Save all monitoring data to file"""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        if not silent:
            print(f"Monitoring data saved to {filename}")

    def export_csv_data(self, filename: str = "fire_data.csv", silent: bool = False):
        """Export key metrics to CSV for analysis"""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 'Changes', 'Fire_Safety_Index', 'Max_Temperature',
                'CO_Concentration', 'Oxygen_Consumed', 'Total_Smoke'
            ])

            for i, stats in enumerate(self.history['statistics']):
                writer.writerow([
                    i,
                    self.history['changes_per_step'][i],
                    stats['fire_safety_index'],
                    stats['max_temperature_celsius'],
                    stats['co_concentration_ppm'],
                    stats['oxygen_consumed_percent'],
                    stats['total_smoke_density']
                ])

        if not silent:
            print(f"CSV data exported to {filename}")


def demo_monitoring():
    """Demonstration of fire monitoring system"""
    print("*** Fire Monitoring System Demo ***")
    print("="*40)

    # Create fire scenario
    rows, cols = 8, 12
    fire_state = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # Add walls
    for i in range(rows):
        fire_state[i][0] = -2
        fire_state[i][cols-1] = -2
    for j in range(cols):
        fire_state[0][j] = -2
        fire_state[rows-1][j] = -2

    # Start fire
    fire_state[2][2] = 1.5

    # Create model and monitor
    model = create_fire_model(rows, cols, wind_speed=1.0, humidity=30.0)
    monitor = FireMonitor(model)

    print("Starting monitored simulation...")

    # Run monitored simulation
    for step in range(6):
        monitoring_data = monitor.monitor_step(fire_state)
        monitor.print_live_monitor(monitoring_data)
        time.sleep(0.5)  # Pause for dramatic effect

    # Generate final report
    print("\n" + "="*50)
    print("FINAL SIMULATION REPORT")
    print("="*50)

    summary = monitor.get_summary_report()
    print(f"Total steps simulated: {summary['simulation_summary']['total_steps']}")
    print(f"Peak activity at step: {summary['simulation_summary']['peak_activity_step']}")
    print(f"Safety classification: {summary['safety_analysis']['safety_classification']}")
    print(f"Max temperature: {summary['environmental_impact']['max_temperature_reached']:.1f}°C")
    print(f"Total oxygen consumed: {summary['environmental_impact']['total_oxygen_consumed']:.1f}%")

    # Save data
    monitor.save_monitoring_data()
    monitor.export_csv_data()


if __name__ == "__main__":
    demo_monitoring()