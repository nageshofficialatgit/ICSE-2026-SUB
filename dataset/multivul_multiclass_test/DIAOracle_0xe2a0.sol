// SPDX-License-Identifier: MIT
pragma solidity >=0.5.0 ^0.8.0;

// src/interfaces/IDIAOracleV2.sol

interface IDIAOracleV2 {
    function getValue(string memory) external view returns (uint128, uint128);
}

// src/interfaces/IOracle.sol

/// @title IOracle
/// @author an IMFer
/// @notice Interface that oracles used by Morpho must implement.
/// @dev It is the user's responsibility to select markets with safe oracles.
interface IOracle {
    /// @notice Returns the price of 1 asset of collateral token quoted in 1 asset of loan token, scaled by 1e36.
    /// @dev It corresponds to the price of 10**(collateral token decimals) assets of collateral token quoted in
    /// 10**(loan token decimals) assets of loan token with `36 + loan token decimals - collateral token decimals`
    /// decimals of precision.
    function price() external view returns (uint256);
}

// src/oracles/DIAOracle.sol

contract DIAOracle is IOracle {
    address public immutable oracle;
    uint256 public immutable decimals;
    string public pool;

    constructor(address _oracle, string memory _pool, uint256 _decimals) {
        require(_oracle != address(0), "Invalid oracle address");
        oracle = _oracle;
        decimals = _decimals;
        pool = _pool;
    }

    function price() external view returns (uint256) {
        (uint256 currPrice,) = IDIAOracleV2(oracle).getValue(pool);

        uint256 oracleScalar = 10 ** (36 - decimals);

        return currPrice * oracleScalar;
    }
}