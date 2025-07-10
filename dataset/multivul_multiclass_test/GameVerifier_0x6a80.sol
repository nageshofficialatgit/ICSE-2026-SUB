pragma solidity 0.8.26;

type Timestamp is uint64;
type GameType is uint32;
type Claim is bytes32;

interface IDisputeGame {
    function gameData() external view returns (GameType gameType_, Claim rootClaim_, bytes memory extraData_);
}

interface IDisputeGameFactory {
    function games(
        GameType _gameType,
        Claim _rootClaim,
        bytes memory _extraData
    )
        external
        view
        returns (IDisputeGame proxy_, Timestamp timestamp_);
}

contract GameVerifier {
    IDisputeGameFactory public immutable disputeGameFactory;

    constructor(IDisputeGameFactory _disputeGameFactory) {
        disputeGameFactory = _disputeGameFactory;
    }

    /// @notice Determines whether a game is registered in the DisputeGameFactory.
    /// @param _game The game to check.
    /// @return Whether the game is factory registered.
    function isGameRegistered(IDisputeGame _game) public view returns (bool) {
        // Grab the game and game data.
        (GameType gameType, Claim rootClaim, bytes memory extraData) = _game.gameData();

        // Grab the verified address of the game based on the game data.
        (IDisputeGame _factoryRegisteredGame,) =
            disputeGameFactory.games({ _gameType: gameType, _rootClaim: rootClaim, _extraData: extraData });

        // Return whether the game is factory registered.
        return address(_factoryRegisteredGame) == address(_game);
    }
}