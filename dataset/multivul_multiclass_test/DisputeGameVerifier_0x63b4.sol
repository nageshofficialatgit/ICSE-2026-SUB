// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

// Redefining types from the monorepo.
type Timestamp is uint64;
type GameType is uint32;
type Claim is bytes32;

/// @title IDisputeGame
/// @notice Stub dispute game interface for the gameData function.
interface IDisputeGame {
    function gameData() external view returns (GameType gameType_, Claim rootClaim_, bytes memory extraData_);
}

/// @title IDisputeGameFactory
/// @notice Stub dispute game interface for the games function.
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

/// @title DisputeGameVerifier
/// @notice Simple contract that verifies that a given dispute game was created by a given factory.
contract DisputeGameVerifier {
    /// @notice Determines whether a game is registered in the DisputeGameFactory.
    /// @param _game The game to check.
    /// @return Whether the game is factory registered.
    function isGameRegistered(
        IDisputeGameFactory _factory,
        IDisputeGame _game
    ) public view returns (bool) {
        // Grab the game and game data.
        (GameType gameType, Claim rootClaim, bytes memory extraData) = _game.gameData();

        // Grab the verified address of the game based on the game data.
        (IDisputeGame _factoryRegisteredGame,) =
            _factory.games({ _gameType: gameType, _rootClaim: rootClaim, _extraData: extraData });

        // Return whether the game is factory registered.
        return address(_factoryRegisteredGame) == address(_game);
    }
}