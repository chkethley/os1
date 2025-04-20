import abc
import uuid
import random
import numpy as np
import copy
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Set

# Import the NeSy components (assuming they're in nesy.py)
from nesy import NeSyUnit, SimpleHybridComputation, SimpleGradientAndFactUpdate
from nesy import PrintBasedCommunication, JsonFileStateManagement

class AgentBlueprint:
    """Defines the structure and capabilities of an agent."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 capabilities: List[str],
                 nesy_unit_configs: List[Dict[str, Any]],
                 meta_parameters: Dict[str, Any] = None):
        """
        Initialize an agent blueprint.
        
        Args:
            name: Unique name for this blueprint type
            description: Human-readable description
            capabilities: List of capability strings this agent type has
            nesy_unit_configs: Configurations for NeSy units this agent will use
            meta_parameters: Additional parameters controlling agent behavior
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.nesy_unit_configs = nesy_unit_configs
        self.meta_parameters = meta_parameters or {}
        self.creation_timestamp = time.time()
        self.version = 1
        self.ancestor_ids = []
        
    def to_dict(self) -> Dict:
        """Convert blueprint to serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "nesy_unit_configs": self.nesy_unit_configs,
            "meta_parameters": self.meta_parameters,
            "creation_timestamp": self.creation_timestamp,
            "version": self.version,
            "ancestor_ids": self.ancestor_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentBlueprint':
        """Create blueprint from dictionary representation."""
        blueprint = cls(
            name=data["name"],
            description=data["description"],
            capabilities=data["capabilities"],
            nesy_unit_configs=data["nesy_unit_configs"],
            meta_parameters=data["meta_parameters"]
        )
        blueprint.id = data["id"]
        blueprint.creation_timestamp = data["creation_timestamp"]
        blueprint.version = data["version"]
        blueprint.ancestor_ids = data["ancestor_ids"]
        return blueprint
    
    def mutate(self, mutation_rate: float = 0.1) -> 'AgentBlueprint':
        """
        Create a mutated version of this blueprint.
        
        Args:
            mutation_rate: Probability of each element being mutated
            
        Returns:
            A new blueprint with mutations
        """
        new_blueprint = copy.deepcopy(self)
        new_blueprint.id = str(uuid.uuid4())
        new_blueprint.ancestor_ids.append(self.id)
        new_blueprint.version = self.version + 1
        new_blueprint.creation_timestamp = time.time()
        
        # Mutate meta parameters
        for key, value in new_blueprint.meta_parameters.items():
            if random.random() < mutation_rate:
                if isinstance(value, bool):
                    new_blueprint.meta_parameters[key] = not value
                elif isinstance(value, (int, float)):
                    # Add random noise between -30% and +30%
                    new_blueprint.meta_parameters[key] = value * (1 + (random.random() - 0.5) * 0.6)
                    if isinstance(value, int):
                        new_blueprint.meta_parameters[key] = int(new_blueprint.meta_parameters[key])
        
        # Mutate NeSy unit configurations
        for unit_config in new_blueprint.nesy_unit_configs:
            if "neural_state" in unit_config and "weights" in unit_config["neural_state"]:
                weights = np.array(unit_config["neural_state"]["weights"])
                
                # Apply small random changes to weights
                if random.random() < mutation_rate:
                    noise = np.random.normal(0, 0.1, weights.shape)
                    weights = weights + noise
                    unit_config["neural_state"]["weights"] = weights.tolist()
            
            # Potentially add or remove a symbolic fact
            if "symbolic_state" in unit_config and "facts" in unit_config["symbolic_state"]:
                facts = set(unit_config["symbolic_state"]["facts"])
                
                if random.random() < mutation_rate and len(facts) > 0:
                    # Remove a random fact
                    fact_to_remove = random.choice(list(facts))
                    facts.remove(fact_to_remove)
                
                unit_config["symbolic_state"]["facts"] = list(facts)
        
        return new_blueprint


class Agent:
    """An intelligent agent composed of NeSy units following a blueprint."""
    
    def __init__(self, 
                 blueprint: AgentBlueprint, 
                 context: Dict[str, Any] = None):
        """
        Initialize an agent from a blueprint.
        
        Args:
            blueprint: The blueprint defining this agent's structure
            context: Optional contextual information for initialization
        """
        self.id = str(uuid.uuid4())
        self.blueprint = blueprint
        self.nesy_units: Dict[str, NeSyUnit] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.creation_time = time.time()
        self.last_active_time = self.creation_time
        self.total_tasks_completed = 0
        self.memory: Dict[str, Any] = {
            "episodic": [],
            "semantic": {},
            "working": {}
        }
        
        # Initialize NeSy units based on blueprint
        self._initialize_nesy_units()
        
    def _initialize_nesy_units(self):
        """Create and configure NeSy units based on blueprint."""
        for i, unit_config in enumerate(self.blueprint.nesy_unit_configs):
            unit_id = unit_config.get("unit_id", f"{self.id}_unit_{i}")
            unit_type = unit_config.get("unit_type", "HybridProcessor")
            
            # Create concrete strategy instances
            comp_strategy = SimpleHybridComputation()
            learn_strategy = SimpleGradientAndFactUpdate()
            comm_protocol = PrintBasedCommunication()
            state_manager = JsonFileStateManagement()
            
            # Get initial states
            neural_state = unit_config.get("neural_state", {})
            symbolic_state = unit_config.get("symbolic_state", {})
            
            # Convert lists to numpy arrays for neural state
            if "weights" in neural_state and isinstance(neural_state["weights"], list):
                neural_state["weights"] = np.array(neural_state["weights"])
            if "bias" in neural_state and isinstance(neural_state["bias"], list):
                neural_state["bias"] = np.array(neural_state["bias"])
                
            # Convert facts list to set for symbolic state
            if "facts" in symbolic_state and isinstance(symbolic_state["facts"], list):
                symbolic_state["facts"] = set(symbolic_state["facts"])
            
            # Create the NeSy unit
            unit = NeSyUnit(
                unit_id=unit_id,
                unit_type=unit_type,
                computation_strategy=comp_strategy,
                learning_strategy=learn_strategy,
                communication_protocol=comm_protocol,
                state_manager=state_manager,
                initial_neural_state=neural_state,
                initial_symbolic_state=symbolic_state,
                configuration=unit_config.get("configuration", {})
            )
            
            self.nesy_units[unit_id] = unit
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using this agent's NeSy units.
        
        Args:
            task: Task description and input data
            
        Returns:
            Task result
        """
        task_id = task.get("id", str(uuid.uuid4()))
        task_type = task.get("type", "unknown")
        task_inputs = task.get("inputs", {})
        
        print(f"Agent {self.id} processing task {task_id} of type {task_type}")
        
        # Update agent state
        self.last_active_time = time.time()
        
        # Record in episodic memory
        memory_entry = {
            "timestamp": self.last_active_time,
            "task_id": task_id,
            "task_type": task_type
        }
        self.memory["episodic"].append(memory_entry)
        
        # Use working memory for task context
        self.memory["working"] = {
            "current_task": task,
            "intermediate_results": {},
            "active_units": []
        }
        
        # Process through NeSy units
        results = {}
        unit_outputs = {}
        
        # Simple sequential processing through all units
        # A more sophisticated agent would determine which units to use based on the task
        for unit_id, unit in self.nesy_units.items():
            self.memory["working"]["active_units"].append(unit_id)
            
            # Prepare inputs for this unit - combine task inputs with outputs from previous units
            unit_inputs = np.array(list(task_inputs.values()))
            
            # Additional context from symbolic requirements
            required_fact = None
            if task_type in self.blueprint.capabilities:
                required_fact = f"can_handle_{task_type}"
            
            # Compute using this unit
            try:
                output = unit.compute(unit_inputs, required_fact=required_fact)
                unit_outputs[unit_id] = output
                
                # Store intermediate result
                self.memory["working"]["intermediate_results"][unit_id] = output
                
                # If output indicates a symbolic constraint failure, try to add the required capability
                if output == "SYMBOLIC_CONSTRAINT_FAILED" and required_fact:
                    print(f"Unit {unit_id} lacks capability: {required_fact}. Attempting to learn...")
                    feedback = {
                        'symbolic_feedback': {
                            'add_fact': required_fact
                        }
                    }
                    unit.update_state(feedback)
                    
                    # Try again
                    output = unit.compute(unit_inputs, required_fact=required_fact)
                    unit_outputs[unit_id] = output
                    self.memory["working"]["intermediate_results"][unit_id] = output
            except Exception as e:
                print(f"Error in unit {unit_id}: {e}")
                unit_outputs[unit_id] = f"ERROR: {str(e)}"
        
        # Combine results (simple approach)
        results = {
            "task_id": task_id,
            "status": "completed",
            "outputs": unit_outputs,
            "processing_time": time.time() - self.last_active_time
        }
        
        # Record task completion
        self.total_tasks_completed += 1
        
        return results
    
    def learn_from_feedback(self, task_id: str, feedback: Dict[str, Any]):
        """
        Update agent based on feedback about a task.
        
        Args:
            task_id: ID of the task the feedback is about
            feedback: Feedback data including success metrics, gradients, etc.
        """
        success_rating = feedback.get("success_rating", 0.0)
        
        # Record performance
        performance_entry = {
            "task_id": task_id,
            "timestamp": time.time(),
            "success_rating": success_rating,
            "feedback": feedback
        }
        self.performance_history.append(performance_entry)
        
        # Update semantic memory with any new knowledge
        if "new_knowledge" in feedback:
            for key, value in feedback["new_knowledge"].items():
                self.memory["semantic"][key] = value
        
        # Update NeSy units
        if "unit_feedback" in feedback:
            for unit_id, unit_feedback in feedback["unit_feedback"].items():
                if unit_id in self.nesy_units:
                    self.nesy_units[unit_id].update_state(unit_feedback)
    
    def get_average_performance(self) -> float:
        """Calculate the agent's average performance score."""
        if not self.performance_history:
            return 0.0
        
        return sum(entry["success_rating"] for entry in self.performance_history) / len(self.performance_history)
    
    def save_state(self, file_path: str = None):
        """
        Save the agent's state to a file.
        
        Args:
            file_path: Optional path to save to, defaults to agent_[id].json
        """
        if file_path is None:
            file_path = f"agent_{self.id}.json"
        
        # Save main agent data
        agent_data = {
            "id": self.id,
            "blueprint_id": self.blueprint.id,
            "blueprint": self.blueprint.to_dict(),
            "creation_time": self.creation_time,
            "last_active_time": self.last_active_time,
            "total_tasks_completed": self.total_tasks_completed,
            "performance_history": self.performance_history,
            "memory": {
                "episodic": self.memory["episodic"],
                "semantic": self.memory["semantic"]
                # Note: Working memory is transient and not saved
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        # Save each NeSy unit separately
        for unit_id, unit in self.nesy_units.items():
            unit.save_state()
        
        print(f"Agent {self.id} state saved to {file_path}")
    
    @classmethod
    def load_state(cls, file_path: str) -> 'Agent':
        """
        Load an agent from a saved state file.
        
        Args:
            file_path: Path to the agent state file
            
        Returns:
            Reconstructed Agent instance
        """
        with open(file_path, 'r') as f:
            agent_data = json.load(f)
        
        # Reconstruct blueprint
        blueprint = AgentBlueprint.from_dict(agent_data["blueprint"])
        
        # Create agent instance
        agent = cls(blueprint)
        
        # Restore agent properties
        agent.id = agent_data["id"]
        agent.creation_time = agent_data["creation_time"]
        agent.last_active_time = agent_data["last_active_time"]
        agent.total_tasks_completed = agent_data["total_tasks_completed"]
        agent.performance_history = agent_data["performance_history"]
        agent.memory["episodic"] = agent_data["memory"]["episodic"]
        agent.memory["semantic"] = agent_data["memory"]["semantic"]
        
        # Load NeSy units
        for unit_id in agent.nesy_units:
            unit_file = f"nesy_unit_{unit_id}_state.json"
            try:
                agent.nesy_units[unit_id].load_state(unit_file)
            except Exception as e:
                print(f"Error loading NeSy unit {unit_id}: {e}")
        
        return agent


class AgentFactory:
    """Creates, manages, and evolves agent blueprints and instances."""
    
    def __init__(self):
        """Initialize the agent factory."""
        self.blueprints: Dict[str, AgentBlueprint] = {}
        self.agents: Dict[str, Agent] = {}
    
    def register_blueprint(self, blueprint: AgentBlueprint):
        """
        Register a blueprint for future use.
        
        Args:
            blueprint: The blueprint to register
        """
        self.blueprints[blueprint.id] = blueprint
        print(f"Registered blueprint {blueprint.id}: {blueprint.name}")
    
    def create_agent(self, blueprint_id: str = None, blueprint: AgentBlueprint = None) -> Agent:
        """
        Create a new agent from a blueprint.
        
        Args:
            blueprint_id: ID of a registered blueprint to use
            blueprint: Or directly provide a blueprint object
            
        Returns:
            New Agent instance
        
        Raises:
            ValueError: If neither blueprint_id nor blueprint is provided, or if blueprint_id doesn't exist
        """
        if blueprint is None:
            if blueprint_id is None:
                raise ValueError("Must provide either blueprint_id or blueprint")
            
            if blueprint_id not in self.blueprints:
                raise ValueError(f"Blueprint with ID {blueprint_id} not found")
            
            blueprint = self.blueprints[blueprint_id]
        
        agent = Agent(blueprint)
        self.agents[agent.id] = agent
        print(f"Created agent {agent.id} using blueprint {blueprint.id}: {blueprint.name}")
        return agent
    
    def evolve_population(self, 
                         population_size: int = 10, 
                         generations: int = 5,
                         selection_pressure: float = 0.5,
                         mutation_rate: float = 0.2,
                         fitness_function: Callable[[Agent], float] = None,
                         initial_blueprint: AgentBlueprint = None,
                         evaluation_tasks: List[Dict[str, Any]] = None):
        """
        Evolve a population of agents through selection and mutation.
        
        Args:
            population_size: Number of agents in each generation
            generations: Number of generations to evolve
            selection_pressure: Fraction of population selected as parents (0-1)
            mutation_rate: Probability of mutation for each component (0-1)
            fitness_function: Function to evaluate agent fitness, defaults to average performance
            initial_blueprint: Starting blueprint, will be randomly initialized if None
            evaluation_tasks: Tasks to evaluate agents on
            
        Returns:
            The most fit agent from the final generation
        """
        if fitness_function is None:
            fitness_function = lambda agent: agent.get_average_performance()
        
        if evaluation_tasks is None:
            # Create simple default evaluation tasks
            evaluation_tasks = [
                {
                    "id": f"task_{i}",
                    "type": "basic_computation",
                    "inputs": {"x1": random.random(), "x2": random.random(), "x3": random.random()}
                }
                for i in range(5)
            ]
        
        print(f"Starting evolutionary process: {population_size} agents, {generations} generations")
        
        # Initialize population with the initial blueprint or create a basic one
        if initial_blueprint is None:
            initial_blueprint = self._create_basic_blueprint()
        
        population = []
        for i in range(population_size):
            if i == 0:
                # First agent uses the initial blueprint directly
                agent = self.create_agent(blueprint=initial_blueprint)
            else:
                # Others are mutations of the initial blueprint
                mutated_blueprint = initial_blueprint.mutate(mutation_rate)
                self.register_blueprint(mutated_blueprint)
                agent = self.create_agent(blueprint=mutated_blueprint)
            
            population.append(agent)
        
        best_agent = None
        best_fitness = float('-inf')
        
        # Run evolutionary process
        for generation in range(generations):
            print(f"\nGeneration {generation+1}/{generations}")
            
            # Evaluate each agent on the tasks
            for agent in population:
                for task in evaluation_tasks:
                    result = agent.process_task(task)
                    
                    # Simple synthetic feedback based on processing time and completion
                    feedback = {
                        "success_rating": 0.5,  # Base score
                        "unit_feedback": {}
                    }
                    
                    # Adjust rating based on processing time (faster is better)
                    processing_time = result.get("processing_time", 1.0)
                    time_factor = max(0, 1.0 - (processing_time / 5.0))  # Normalize 0-5 seconds to 1-0
                    feedback["success_rating"] += time_factor * 0.5
                    
                    # Random variation to simulate real-world performance differences
                    feedback["success_rating"] *= random.uniform(0.8, 1.2)
                    feedback["success_rating"] = min(1.0, max(0.0, feedback["success_rating"]))
                    
                    # Apply feedback to agent
                    agent.learn_from_feedback(task["id"], feedback)
            
            # Calculate fitness for each agent
            agent_fitness = [(agent, fitness_function(agent)) for agent in population]
            agent_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Track best agent
            current_best_agent, current_best_fitness = agent_fitness[0]
            print(f"Best agent this generation: {current_best_agent.id} with fitness {current_best_fitness:.4f}")
            
            if current_best_fitness > best_fitness:
                best_agent = current_best_agent
                best_fitness = current_best_fitness
            
            # Stop if this is the last generation
            if generation == generations - 1:
                break
            
            # Select parents for next generation
            num_parents = max(2, int(population_size * selection_pressure))
            parents = [agent for agent, _ in agent_fitness[:num_parents]]
            
            # Create next generation
            next_population = []
            
            # Keep the best agent unchanged (elitism)
            next_population.append(current_best_agent)
            
            # Create the rest through mutation of parents
            while len(next_population) < population_size:
                parent = random.choice(parents)
                mutated_blueprint = parent.blueprint.mutate(mutation_rate)
                self.register_blueprint(mutated_blueprint)
                new_agent = self.create_agent(blueprint=mutated_blueprint)
                next_population.append(new_agent)
            
            population = next_population
        
        print(f"\nEvolution complete. Best agent: {best_agent.id} with fitness {best_fitness:.4f}")
        return best_agent
    
    def _create_basic_blueprint(self) -> AgentBlueprint:
        """Create a basic blueprint for initialization."""
        # Simple Neural Network with 3 inputs, 2 hidden, 1 output
        input_size = 3
        hidden_size = 2
        output_size = 1
        
        # Random weights initialization
        input_weights = np.random.randn(input_size, hidden_size).tolist()
        hidden_weights = np.random.randn(hidden_size, output_size).tolist()
        
        blueprint = AgentBlueprint(
            name="BasicAgent",
            description="Simple agent with 2 NeSy units",
            capabilities=["basic_computation", "data_processing"],
            nesy_unit_configs=[
                {
                    "unit_id": "input_processor",
                    "unit_type": "InputProcessor",
                    "neural_state": {
                        "weights": input_weights,
                        "bias": np.zeros(hidden_size).tolist()
                    },
                    "symbolic_state": {
                        "facts": ["is_active", "can_handle_basic_computation"]
                    },
                    "configuration": {
                        "activation": "sigmoid",
                        "learning_rate": 0.01
                    }
                },
                {
                    "unit_id": "output_processor",
                    "unit_type": "OutputProcessor",
                    "neural_state": {
                        "weights": hidden_weights,
                        "bias": np.zeros(output_size).tolist()
                    },
                    "symbolic_state": {
                        "facts": ["is_active", "can_handle_data_processing"]
                    },
                    "configuration": {
                        "activation": "sigmoid",
                        "learning_rate": 0.01
                    }
                }
            ],
            meta_parameters={
                "use_working_memory": True,
                "reflection_enabled": False,
                "default_learning_rate": 0.01,
                "symbolic_threshold": 0.7
            }
        )
        
        self.register_blueprint(blueprint)
        return blueprint


class MetaAgent:
    """
    Orchestrates the entire ecosystem, managing agent creation,
    task assignment, and evolution.
    """
    
    def __init__(self):
        """Initialize the meta-agent."""
        self.agent_factory = AgentFactory()
        self.active_agents: Dict[str, Agent] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "task_completion_times": [],
            "success_ratings": [],
            "agent_fitness_scores": []
        }
        
        # Memory system
        self.global_memory = {
            "facts": {},                # Semantic memory
            "episodes": [],             # Episodic memory
            "skills": {},               # Procedural memory
            "agent_performance": {}     # Meta-memory about agent performance
        }
    
    def register_blueprint(self, blueprint: AgentBlueprint):
        """Register a blueprint with the agent factory."""
        self.agent_factory.register_blueprint(blueprint)
    
    def create_agent(self, blueprint_id: str = None, blueprint: AgentBlueprint = None) -> str:
        """
        Create a new agent and add it to active agents.
        
        Returns:
            ID of the new agent
        """
        agent = self.agent_factory.create_agent(blueprint_id, blueprint)
        self.active_agents[agent.id] = agent
        return agent.id
    
    def evolve_agent_population(self, task_types: List[str], population_size: int = 10, generations: int = 5) -> str:
        """
        Evolve a population of agents specialized for specific task types.
        
        Args:
            task_types: List of task types to optimize for
            population_size: Size of the agent population
            generations: Number of generations to evolve
            
        Returns:
            ID of the best evolved agent
        """
        # Create evaluation tasks based on requested types
        evaluation_tasks = []
        for task_type in task_types:
            for i in range(3):  # Create 3 variants of each task type
                evaluation_tasks.append({
                    "id": f"{task_type}_task_{i}",
                    "type": task_type,
                    "inputs": {
                        "x1": random.random(), 
                        "x2": random.random(), 
                        "x3": random.random()
                    },
                    "difficulty": random.uniform(0.3, 1.0)
                })
        
        # Create a blueprint with the requested capabilities
        initial_blueprint = self._create_specialized_blueprint(task_types)
        self.register_blueprint(initial_blueprint)
        
        # Run evolution
        best_agent = self.agent_factory.evolve_population(
            population_size=population_size,
            generations=generations,
            initial_blueprint=initial_blueprint,
            evaluation_tasks=evaluation_tasks
        )
        
        # Add to active agents
        self.active_agents[best_agent.id] = best_agent
        
        # Record in global memory
        self.global_memory["agent_performance"][best_agent.id] = {
            "specializations": task_types,
            "avg_performance": best_agent.get_average_performance(),
            "tasks_completed": best_agent.total_tasks_completed
        }
        
        return best_agent.id
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task by selecting the most appropriate agent.
        
        Args:
            task: Task to process
            
        Returns:
            Task results
        """
        task_id = task.get("id", str(uuid.uuid4()))
        task_type = task.get("type", "unknown")
        
        print(f"Meta-Agent processing task {task_id} of type {task_type}")
        
        # Record task in history
        task_record = {
            "task_id": task_id,
            "task_type": task_type,
            "start_time": time.time(),
            "status": "assigned"
        }
        self.task_history.append(task_record)
        
        # Select best agent for this task
        agent_id = self._select_agent_for_task(task)
        
        if not agent_id:
            print(f"No suitable agent found for task {task_id}. Creating specialized agent...")
            agent_id = self.evolve_agent_population([task_type], population_size=5, generations=3)
        
        # Update task record
        task_record["assigned_agent"] = agent_id
        
        # Process task with selected agent
        agent = self.active_agents[agent_id]
        result = agent.process_task(task)
        
        # Update task record
        task_record["end_time"] = time.time()
        task_record["duration"] = task_record["end_time"] - task_record["start_time"]
        task_record["status"] = "completed"
        task_record["result"] = result
        
        # Update performance metrics
        self.performance_metrics["task_completion_times"].append(task_record["duration"])
        
        # Generate feedback based on results
        feedback = self._generate_feedback(task, result)
        agent.learn_from_feedback(task_id, feedback)
        
        self.performance_metrics["success_ratings"].append(feedback["success_rating"])
        
        # Update global memory
        self.global_memory["episodes"].append({
            "task_id": task_id,
            "task_type": task_type,
            "agent_id": agent_id,
            "success_rating": feedback["success_rating"],
            "timestamp": time.time()
        })
        
        if feedback["success_rating"] > 0.7:
            # Record successful approach in procedural memory
            self.global_memory["skills"][task_type] = {
                "best_agent_id": agent_id,
                "last_success_time": time.time(),
                "success_rating": feedback["success_rating"]
            }
        
        # Update agent performance tracking
        if agent_id in self.global_memory["agent_performance"]:
            self.global_memory["agent_performance"][agent_id]["tasks_completed"] += 1
            
            # Update running average
            current = self.global_memory["agent_performance"][agent_id]
            old_avg = current["avg_performance"]
            old_count = current["tasks_completed"] - 1  # We already incremented above
            new_rating = feedback["success_rating"]
            new_avg = (old_avg * old_count + new_rating) / current["tasks_completed"]
            current["avg_performance"] = new_avg
        
        return result
    
    def _select_agent_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate agent for a task.
        
        Args:
            task: Task to be processed
            
        Returns:
            ID of selected agent, or None if no suitable agent is found
        """
        task_type = task.get("type", "unknown")
        
        # Check if we have a successful agent for this task type in our skills memory
        if task_type in self.global_memory["skills"]:
            skill_info = self.global_memory["skills"][task_type]
            best_agent_id = skill_info["best_agent_id"]
            
            # Verify agent is still available
            if best_agent_id in self.active_agents:
                print(f"Selected agent {best_agent_id} based on past success with {task_type} tasks")
                return best_agent_id
        
        # If no specific skill record, check all active agents for capability
        capable_agents = []
        for agent_id, agent in self.active_agents.items():
            if task_type in agent.blueprint.capabilities:
                # Calculate a fitness score based on past performance
                avg_performance = agent.get_average_performance()
                
                # Bonus for specialization in this task type
                specialization_bonus = 0.2 if task_type in agent.blueprint.capabilities else 0
                
                # Penalty for being very busy (not implemented in this prototype)
                busyness_penalty = 0
                
                fitness = avg_performance + specialization_bonus - busyness_penalty
                capable_agents.append((agent_id, fitness))
        
        # Sort by fitness
        capable_agents.sort(key=lambda x: x[1], reverse=True)
        
        if capable_agents:
            selected_agent_id, fitness = capable_agents[0]
            print(f"Selected agent {selected_agent_id} with fitness {fitness:.4f} for task {task['id']}")
            return selected_agent_id
        
        return None
    
    def _generate_feedback(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate feedback for a completed task.
        
        Args:
            task: Original task
            result: Task result
            
        Returns:
            Feedback data
        """
        # In a real system, this would involve actual evaluation of results
        # For this prototype, we use processing time and simulate success
        
        processing_time = result.get("processing_time", 1.0)
        
        # Basic rating: Faster is better
        success_rating = max(0, 1.0 - (processing_time / 5.0))
        
        # Simulate some variability
        success_rating = min(1.0, max(0.1, success_rating * random.uniform(0.8, 1.2)))
        
        feedback = {
            "success_rating": success_rating,
            "processing_time": processing_time,
            "unit_feedback": {}
        }
        
        return feedback
    
    def _create_specialized_blueprint(self, capabilities: List[str]) -> AgentBlueprint:
        """
        Create a blueprint specialized for specific capabilities.
        
        Args:
            capabilities: List of capabilities to specialize for
            
        Returns:
            Specialized blueprint
        """
        input_size = 3
        hidden_size = 4
        output_size = 1
        
        # Random weights initialization
        input_weights = np.random.randn(input_size, hidden_size).tolist()
        hidden_weights = np.random.randn(hidden_size, output_size).tolist()
        
        # Create symbolic facts for each capability
        capability_facts = ["is_active"]
        for capability in capabilities:
            capability_facts.append(f"can_handle_{capability}")
        
        blueprint = AgentBlueprint(
            name=f"Specialized_{'_'.join(capabilities)}",
            description=f"Agent specialized for: {', '.join(capabilities)}",
            capabilities=capabilities,
            nesy_unit_configs=[
                {
                    "unit_id": "specialization_unit",
                    "unit_type": "SpecializationProcessor",
                    "neural_state": {
                        "weights": input_weights,
                        "bias": np.zeros(hidden_size).tolist()
                    },
                    "symbolic_state": {
                        "facts": capability_facts
                    },
                    "configuration": {
                        "activation": "sigmoid",
                        "learning_rate": 0.01,
                        "specialization": capabilities[0] if capabilities else "general"
                    }
                },
                {
                    "unit_id": "output_processor",
                    "unit_type": "OutputProcessor",
                    "neural_state": {
                        "weights": hidden_weights,
                        "bias": np.zeros(output_size).tolist()
                    },
                    "symbolic_state": {
                        "facts": ["is_active"]
                    },
                    "configuration": {
                        "activation": "sigmoid",
                        "learning_rate": 0.01
                    }
                }
            ],
            meta_parameters={
                "use_working_memory": True,
                "reflection_enabled": True,
                "default_learning_rate": 0.02,
                "symbolic_threshold": 0.6,
                "specialization_boost": 0.3
            }
        )
        
        return blueprint