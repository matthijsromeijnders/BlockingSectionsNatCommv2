
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class Service:
    agent: int
    start_end: Tuple[float, float]
    event_ids: List[int]
    event_blocks: List[int] 

    # Primary role
    primary_partners: List[int] = field(default_factory=list)
    primary_event_ids: List[List[int]] = field(default_factory=list)
    primary_event_blocks: List[List[int]] = field(default_factory=list)
    primary_start_ends: List[Tuple[float, float]] = field(default_factory=list)
    primary_buffers: List[List[float]] = field(default_factory=list)
    primary_min_buffers: List[float] = field(default_factory=list)

    # Secondary role
    secondary_partners: List[int] = field(default_factory=list)
    secondary_event_ids: List[List[int]] = field(default_factory=list)
    secondary_event_blocks: List[List[int]] = field(default_factory=list)
    secondary_start_ends: List[Tuple[float, float]] = field(default_factory=list)
    secondary_buffers: List[List[float]] = field(default_factory=list)
    secondary_min_buffers: List[float] = field(default_factory=list)


@dataclass
class ServiceSheet:
    services: Dict[int, Service] = field(default_factory=dict)

    @staticmethod
    def from_event_data(agents: List[int], event_arr: np.ndarray, buffer_arr: np.ndarray) -> "ServiceSheet":
        sheet = ServiceSheet()
        sheet.event_arr = event_arr
        sheet.buffer_arr = buffer_arr
        for agent in agents:
            agent = int(agent)
            events_prim = np.where(event_arr[:, 1] == agent)[0]
            all_events = np.where((event_arr[:, 1] == agent) | (event_arr[:, 2] == agent))[0]
            # print(all_events)
            start_end = (event_arr[all_events[0], 3], event_arr[all_events[-1], 3])
            event_blocks = event_arr[all_events, 4].tolist()
            sheet.services[agent] = Service(agent=agent, start_end=start_end, event_ids=all_events.tolist(), event_blocks=event_blocks)

        for agent, service in sheet.services.items():
            # Previously 'as primary' → now assign to 'secondary'
            for j in np.unique(event_arr[:, 2][event_arr[:, 1] == agent]):
                if agent == j: continue
                joint_events = np.where((event_arr[:, 1] == agent) & (event_arr[:, 2] == j))[0]
                buffers = buffer_arr[joint_events].tolist()
                service.secondary_partners.append(j)
                service.secondary_event_ids.append(joint_events.tolist())
                service.secondary_start_ends.append((event_arr[joint_events[0], 3], event_arr[joint_events[-1], 3]))
                service.secondary_buffers.append(buffers)
                service.secondary_min_buffers.append(min(buffers))
                service.secondary_event_blocks.append(event_arr[joint_events, 4].tolist())
            # Previously 'as secondary' → now assign to 'primary'
            for i in np.unique(event_arr[:, 1][event_arr[:, 2] == agent]):
                if agent == i: continue
                joint_events = np.where((event_arr[:, 1] == i) & (event_arr[:, 2] == agent))[0]
                buffers = buffer_arr[joint_events].tolist()
                service.primary_partners.append(i)
                service.primary_event_ids.append(joint_events.tolist())
                service.primary_start_ends.append((event_arr[joint_events[0], 3], event_arr[joint_events[-1], 3]))
                service.primary_buffers.append(buffers)
                service.primary_min_buffers.append(min(buffers))
                service.primary_event_blocks.append(event_arr[joint_events, 4].tolist())
        return sheet

    @staticmethod
    def from_df(df: pd.DataFrame, event_dict = None) -> "ServiceSheet":
        uniI = np.unique(df["i"].to_numpy().astype(int))
        uniJ = np.unique(df["j"].to_numpy().astype(int))
        agents = np.unique(np.concatenate((uniI, uniJ)))
        lenevents = len(np.unique(df.event_id))
        event_arr = np.zeros((lenevents, 5)).astype(int)
        buffer_arr = np.zeros((len(df.buffer)))
        for _, row in df.iterrows():
            i = row['i']
            j = row['j']
            t = row['t']
            buffer = row["buffer"]
            event_id = row['event_id']
            block_id = row['block']
            event_arr[int(event_id), :] = np.array([int(event_id), int(i), int(j), int(t), int(block_id)])
            buffer_arr[int(event_id)] = float(buffer)    
        sheet = ServiceSheet.from_event_data(agents, event_arr, buffer_arr)
        return sheet
    
    def get(self, agent: int) -> Service:
        return self.services[agent]

    def get_next_event_id(self, agent: int, event_id: int) -> Tuple[int]:
        s = self.services[agent]
        if event_id in s.event_ids:
            idx = s.event_ids.index(event_id)
            if idx == len(s.event_ids) - 1:
                return event_id
            else:
                return s.event_ids[idx + 1]
        else:
            raise ValueError(f"Event ID {event_id} not found for agent {agent}")

    def list_all_agents(self) -> List[int]:
        return list(self.services.keys())

    def describe_agent_services(self, agent: int) -> Dict:
        s = self.services[agent]
        return {
            "primary_partners": s.primary_partners,
            "secondary_partners": s.secondary_partners,
            "start_end": s.start_end
        }

    def get_min_buffer_between(self, agent1: int, agent2: int) -> float:
        s1 = self.services[agent1]
        if agent2 in s1.primary_partners:
            idx = s1.primary_partners.index(agent2)
            return s1.primary_min_buffers[idx]
        if agent2 in s1.secondary_partners:
            idx = s1.secondary_partners.index(agent2)
            return s1.secondary_min_buffers[idx]
        raise ValueError(f"No interaction found between agents {agent1} and {agent2}")

    def get_all_agent_start_ends(self) -> Dict[int, Tuple[float, float]]:
        return {agent: s.start_end for agent, s in self.services.items()}

    def get_all_pairwise_min_buffers(self) -> Dict[Tuple[int, int], float]:
        result = {}
        for agent, s in self.services.items():
            for i, partner in enumerate(s.primary_partners):
                result[(agent, partner)] = s.primary_min_buffers[i]
            for i, partner in enumerate(s.secondary_partners):
                result[(partner, agent)] = s.secondary_min_buffers[i]
        return result

    def get_total_min_buffers(self) -> float:
        return sum(self.get_all_pairwise_min_buffers().values())

    def get_average_min_buffers(self) -> float:
        vals = list(self.get_all_pairwise_min_buffers().values())
        return float(np.mean(vals)) if vals else 0.0

    def get_all_min_buffers(self) -> List[float]:
        return list(self.get_all_pairwise_min_buffers().values())

    def get_agent_partner_stats(self) -> Dict[str, float]:
        primary_counts = [len(s.primary_partners) for s in self.services.values()]
        secondary_counts = [len(s.secondary_partners) for s in self.services.values()]
        return {
            "primary_avg": float(np.mean(primary_counts)),
            "primary_min": int(np.min(primary_counts)),
            "primary_max": int(np.max(primary_counts)),
            "primary_std": float(np.std(primary_counts)),
            "secondary_avg": float(np.mean(secondary_counts)),
            "secondary_min": int(np.min(secondary_counts)),
            "secondary_max": int(np.max(secondary_counts)),
            "secondary_std": float(np.std(secondary_counts)),
        }

    def build_directional_graph(self, subset: List[int] = None) -> nx.DiGraph:
        G = nx.DiGraph()
        subset_set = set(subset) if subset is not None else None

        for agent, service in self.services.items():
            if subset_set and agent not in subset_set:
                continue
            for i, secondary in enumerate(service.secondary_partners):
                if subset_set and secondary not in subset_set:
                    continue
                G.add_edge(agent, secondary, weight=service.secondary_min_buffers[i])
        return G

    def plot_graph(self, G: nx.DiGraph = None, subset: List[int] = None, with_labels: bool = True, node_size: int = 700):
        if G is None:
            G = self.build_directional_graph(subset=subset)
        pos = nx.spring_layout(G)
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=with_labels, node_size=node_size, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()})
        plt.title("Service Interaction Graph")
        plt.show()

    def _compute_hierarchy_levels(self) -> Dict[int, int]:
        from collections import deque, defaultdict

        levels = {}
        roots = [a for a, s in self.services.items() if not s.primary_partners]
        queue = deque([(r, 0) for r in roots])
        visited = set(roots)

        for r in roots:
            levels[r] = 0

        while queue:
            current, depth = queue.popleft()
            for partner in self.services[current].secondary_partners:
                if partner not in levels or levels[partner] > depth + 1:
                    levels[partner] = depth + 1
                    if partner not in visited:
                        queue.append((partner, depth + 1))
                        visited.add(partner)

        return levels


    def plot_hierarchical_graph(self, subset: List[int] = None, node_size: int = 700, with_edge_labels=False):
        G = self.build_directional_graph(subset=subset)
        levels = self._compute_hierarchy_levels()

        if subset is not None:
            levels = {k: v for k, v in levels.items() if k in subset}
            G = G.subgraph(subset)

        from collections import defaultdict
        level_to_nodes = defaultdict(list)
        for node, level in levels.items():
            level_to_nodes[level].append(node)

        pos = {}
        for y, level in enumerate(sorted(level_to_nodes.keys())):
            nodes = sorted(level_to_nodes[level])
            for x, node in enumerate(nodes):
                pos[node] = (y, x)

        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_size=node_size, arrows=True)
        if with_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()})
        else:
            nx.draw_networkx_edge_labels(G, pos, edge_labels={})
        plt.title("Hierarchical Service Graph")
        plt.show()

    def print_service_schedule(self, agent: int):
        df = self.get_service_timetable(agent)
        lines = [f"Agent {agent}:"]
        for _, row in df.iterrows():
            if row["action"] == "start":
                lines.append(f"  Start: event {row.event_id} at time {row.time:.2f}")
            elif row["action"] == "end":
                lines.append(f"  End: event {row.event_id} at time {row.time:.2f}")
            elif row["role"] == "primary":
                lines.append(f"  → Switch to primary of {row.partner} at event {row.event_id}, time {row.time:.2f}")
            elif row["role"] == "secondary":
                lines.append(f"  → Switch to secondary of {row.partner} at event {row.event_id}, time {row.time:.2f}")
        print("\n".join(lines))


    def _get_event_time(self, event_id: int) -> float:
        # You must ensure `self.event_arr` is available in the ServiceSheet
        return self.event_arr[event_id, 3]


    def get_service_timetable(self, agent: int) -> pd.DataFrame:
        s = self.services[agent]
        rows = []

        # 1. Build unsorted event rows
        rows.append({
            "agent": agent,
            "event_id": s.event_ids[0],
            "time": s.start_end[0],
            "action": "start",
            "partner": None,
            "role": None,
        })

        for partner, evs in zip(s.secondary_partners, s.secondary_event_ids):
            rows.append({
                "agent": agent,
                "event_id": evs[0],
                "time": self._get_event_time(evs[0]),
                "action": "switch",
                "partner": partner,
                "role": "secondary",
            })

        for partner, evs in zip(s.primary_partners, s.primary_event_ids):
            rows.append({
                "agent": agent,
                "event_id": evs[0],
                "time": self._get_event_time(evs[0]),
                "action": "switch",
                "partner": partner,
                "role": "primary",
            })

        rows.append({
            "agent": agent,
            "event_id": s.event_ids[-1],
            "time": s.start_end[1],
            "action": "end",
            "partner": None,
            "role": None,
        })

        # 2. Sort chronologically
        df = pd.DataFrame(rows).sort_values(by="time").reset_index(drop=True)

        # 3. Track live state for each row
        agent_ahead = None
        agent_behind = None
        buffer_ahead = None
        buffer_behind = None

        ahead_list = []
        behind_list = []
        buf_ahead_list = []
        buf_behind_list = []

        for _, row in df.iterrows():
            if row["role"] == "primary" and row["partner"] is not None:
                partner = row["partner"]
                partner_service = self.services[partner]
                if agent in partner_service.secondary_partners:
                    idx = partner_service.secondary_partners.index(agent)
                    buffer_ahead = partner_service.secondary_min_buffers[idx]
                    agent_ahead = partner

            elif row["role"] == "secondary" and row["partner"] is not None:
                partner = row["partner"]
                if partner in s.secondary_partners:
                    idx = s.secondary_partners.index(partner)
                    buffer_behind = s.secondary_min_buffers[idx]
                    agent_behind = partner

            ahead_list.append(agent_ahead)
            behind_list.append(agent_behind)
            buf_ahead_list.append(buffer_ahead)
            buf_behind_list.append(buffer_behind)

        df["agent_ahead"] = ahead_list
        df["agent_behind"] = behind_list
        df["buffer_ahead"] = buf_ahead_list
        df["buffer_behind"] = buf_behind_list

        # 4. Compute start_block and end_block for each row
        block_map = dict(zip(s.event_ids, s.event_blocks))
        start_blocks = []
        end_blocks = []

        for _, row in df.iterrows():
            if row["action"] == "start":
                start_blocks.append(block_map[row["event_id"]])
                end_blocks.append(block_map[row["event_id"]])
            elif row["action"] == "end":
                start_blocks.append(block_map[row["event_id"]] )
                end_blocks.append(block_map[row["event_id"]])
            else:
                partner = row["partner"]
                try:
                    partner_idx = s.primary_partners.index(partner) 
                    blocks = s.primary_event_blocks[partner_idx]
                    start_blocks.append(blocks[0])
                    end_blocks.append(blocks[-1])
                except ValueError:
                    partner_idx = s.secondary_partners.index(partner) 
                    blocks = s.secondary_event_blocks[partner_idx]
                    start_blocks.append(blocks[0])
                    end_blocks.append(blocks[-1])

        df["start_block"] = start_blocks
        df["end_block"] = end_blocks
        return df
    
    def calculate_primary_buffer_slack(self, agent: int) -> float:
        schedule_df = self.get_service_timetable(agent)
        bahead = schedule_df["buffer_ahead"].to_numpy()
        bbehind = schedule_df["buffer_behind"].to_numpy()
        
        bahead = np.where(bahead == None, np.nan, bahead).astype(str).astype(float)
        bbehind = np.where(bbehind == None, np.nan, bbehind).astype(str).astype(float)
        
        bahead = bahead[~np.isnan(bahead)]
        bbehind = bbehind[~np.isnan(bbehind)]
        
        if len(bahead) == 0:
            slack_ahead = float('inf')
        else:
            slack_ahead= bahead.min()
        
        if len(bbehind) == 0:
            slack_behind = float('inf')
        else:
            slack_behind = bbehind.min()

        return slack_ahead, slack_behind
    
    def calculate_all_primary_buffer_slacks(self):
        all_slacks = np.zeros((len(self.services), 2))
        for i, agent in enumerate(self.services.keys()):
            slack_ahead, slack_behind = self.calculate_primary_buffer_slack(agent)
            all_slacks[i, 0] = slack_ahead
            all_slacks[i, 1] = slack_behind
        self.buffer_slacks = all_slacks

    # PRINT METHODS
    def print_total_min_buffers(self):
        print("Total min buffer sum:", self.get_total_min_buffers())

    def print_average_min_buffers(self):
        print("Average min buffer:", self.get_average_min_buffers())

    def print_partner_stats(self):
        stats = self.get_agent_partner_stats()
        print("Primary agent stats:")
        print(f"  Avg: {stats['primary_avg']}, Min: {stats['primary_min']}, Max: {stats['primary_max']}, Std: {stats['primary_std']}")
        print("Secondary agent stats:")
        print(f"  Avg: {stats['secondary_avg']}, Min: {stats['secondary_min']}, Max: {stats['secondary_max']}, Std: {stats['secondary_std']}")
    
    # PRINT METHODS
    def print_all_agents(self):
        print("Agents:", self.list_all_agents())

    def print_agent_services(self, agent: int):
        desc = self.describe_agent_services(agent)
        print(f"Agent {agent} services:")
        print(f"  Primary partners: {desc['primary_partners']}")
        print(f"  Secondary partners: {desc['secondary_partners']}")
        print(f"  Start-End: {desc['start_end']}")

    def print_min_buffer_between(self, agent1: int, agent2: int):
        val = self.get_min_buffer_between(agent1, agent2)
        print(f"Min buffer between {agent1} and {agent2}: {val}")

    def print_all_agent_start_ends(self):
        d = self.get_all_agent_start_ends()
        print("Start and end times per agent:")
        for agent, se in d.items():
            print(f"  Agent {agent}: start={se[0]}, end={se[1]}")

    def print_all_pairwise_min_buffers(self):
        d = self.get_all_pairwise_min_buffers()
        print("Pairwise minimum buffers:")
        for pair, val in d.items():
            print(f"  {pair}: {val}")

