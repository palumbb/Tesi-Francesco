from flwr.server.server import Server
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, Scalar
from typing import Optional, Tuple, Dict
from strategy import FedQualStrategy
from flwr.common.logger import log
from logging import INFO, DEBUG
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import FitResultsAndFailures, Server, fit_clients

class FedQualServer(Server):
    """Implement server for FedQual."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[FedQualStrategy] = None,
        quality_exclusion: bool,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FedQualStrategy = strategy if strategy is not None else FedQualStrategy()
        self.quality_exclusion = quality_exclusion

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated training using FedQual."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None

        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results using FedQual's aggregation
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit_threshold(
            server_round=server_round, results=results, failures=failures
        ) if self.quality_exclusion else self.strategy.aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Perform evaluation after the training round if needed."""
        return super().evaluate_round(
            server_round=server_round,
            timeout=timeout,
        )
