def calculate_bipolarization(self, topic_id=0):
    topic = self.topic_space.get_topic(topic_id)
    opinions = [agent.calculate_opinion(topic) for agent in self.agents]
    n = len(opinions)
    
    if n == 0:
        return 0.0
    
    d_bar = 0
    for i in range(n):
        for j in range(n):
            d_bar += abs(opinions[i] - opinions[j])
    d_bar /= (n * n)
    
    P_t = 0
    for i in range(n):
        for j in range(n):
            diff = abs(opinions[i] - opinions[j])
            P_t += (diff - d_bar) ** 2
    
    P_t = 4 * P_t / (n * n)
    return P_t