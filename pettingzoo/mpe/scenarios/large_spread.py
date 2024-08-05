import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self, groups, cooperative=False, shuffle_obs=False, sparsity=100000):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = sum(groups)
        num_landmarks = len(groups)
        world.collaborative = True

        self.shuffle_obs = shuffle_obs

        self.cooperative = cooperative
        self.groups = groups
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]
        # generate colors:
        self.colors = [np.random.random(3) for _ in groups]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world
    
        # sparsity radius
        self.sparsity = sparsity

    def reset_world(self, world, np_random):
        # random properties for agents

        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[i]

        # random properties for landmarks
        for landmark, color in zip(world.landmarks, self.colors):
            landmark.color = color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        i = world.agents.index(agent)
        rew = -np.sqrt(
            np.sum(
                np.square(
                    agent.state.p_pos
                    - world.landmarks[self.group_indices[i]].state.p_pos
                )
            )
        )

        # reward sparsity radius around the landmark
        sparsity_radius = self.sparsity
        rew = max(rew, -sparsity_radius)

        if self.cooperative:
            return 0
        else:
            return rew

    def global_reward(self, world):
        rew = 0

        for i, a in zip(self.group_indices, world.agents):
            l = world.landmarks[i]
            rew -= np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))

        if self.cooperative:
            return rew
        else:
            return 0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        # )
        x = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)
        if self.shuffle_obs:
            x = list(x)
            random.Random(self.group_indices[world.agents.index(agent)]).shuffle(x)
            x = np.array(x)
        return x

