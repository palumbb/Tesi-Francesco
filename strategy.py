"""FedNova, SCAFFOLD and FedQual strategies."""

from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

class FedNovaStrategy(FedAvg):
    """Custom FedAvg strategy with fednova based configuration and aggregation."""

    def aggregate_fit_custom(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c_fact = sum(
            [
                float(fit_res.metrics["a_i"]) * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c_fact * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        # Aggregate grad updates, t_eff*(sum_i(p_i*\eta*d_i))
        grad_updates_aggregated = aggregate_fednova(new_weights_results)
        # Final parameters = server_params - grad_updates_aggregated
        aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def aggregate_fednova(results: List[Tuple[NDArrays, float]]) -> NDArrays:
    """Implement custom aggregate function for FedNova."""
    # Create a list of weights, each multiplied by the weight_factor
    weighted_weights = [
        [layer * factor for layer in weights] for weights, factor in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )
    
class FedQualStrategy(FedAvg):
    """Custom FedAvg strategy with quality-weighted aggregation."""

    """def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        Aggregate fit results using quality-weighted average.
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics["quality_weight"])
                for _, fit_res in results
            ]

        # Does in-place weighted average of results
        aggregated_ndarrays = aggregate_fedqual(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.metrics["quality_weight"], res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated"""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results considering quality weights."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        self.inplace = False
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results, aggregate normally without considering num_examples
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics.get("quality_weight", 1.0))
                for _, fit_res in results
            ]
            total_quality_weight = sum(weight for _, weight in weights_results)
            aggregated_ndarrays = [
                sum((weight/total_quality_weight) * param[i] for param, weight in weights_results)
                for i in range(len(weights_results[0][0]))
            ]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate quality metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(1, res.metrics.get("quality_metrics", [])) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    

"""def aggregate_fedqual(results: list[tuple[NDArrays, int]]) -> NDArrays:
    Compute weighted average.

    quality_weights = [ fit_res.metrics["quality_weight"] for _, fit_res in results]
    tot_quality = sum(quality_weights)

        
    final_weights = np.asarray(
        [ parameters_to_ndarrays(param) * (weight/tot_quality) for param, weight in zip(results, quality_weights)]
    )

    # TO FIX 
    return final_weights

    def aggregate_fedqual(results: list[tuple[NDArrays, int]]) -> NDArrays:

    quality_weights = [ w for _, w in results]
    tot_quality = sum(quality_weights)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * (w/tot_quality) for layer in params] for params, w in results
    ]

    new_weights_results = [
            (result[0], w/tot_quality)
            for result, w in results
        ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / 1
        for layer_updates in zip(*new_weights_results)
    ]

    return new_weights_results"""