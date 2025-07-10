// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

// =============================================================================
contract ExcessivelySafeCall {
    struct Call {
        uint96 returnDataSizeLimit; // uint96 + address = word
        address to;
        uint256 value;
        uint256 gasLimit;
        bytes input;
    }

    struct Result {
        bool success;
        bytes returnData;
    }

    // =========================================================================
    function multiSafeCall(Call[] calldata callArray)
    external payable
    returns (Result[] memory resultArray) {
        uint256 length = callArray.length;
        resultArray = new Result[](length);
        for (uint256 i = 0; i < length;) {
            Call memory call = callArray[i];
            (bool success, bytes memory returnData) = safeCall(
                call.returnDataSizeLimit,
                call.to,
                call.value,
                call.gasLimit,
                call.input
            );
            resultArray[i] = Result(success, returnData);
            unchecked {i++;}
        }
        return resultArray;
    }

    // -------------------------------------------------------------------------
    // change from https://github.com/nomad-xyz/ExcessivelySafeCall/blob/main/src/ExcessivelySafeCall.sol#L8C5-L60C6
    function safeCall(
        uint96 returnDataSizeLimit,
        address to,
        uint256 value,
        uint256 gasLimit,
        bytes memory input
    )
    public payable
    returns (bool, bytes memory) {
        bool success;
        uint256 returnDataSize;

        assembly {
            success := call(
                gasLimit, // gas
                to, // address
                value, // value
                add(input, 0x20), // argsOffset
                mload(input), // argsSize
                0, // retOffset
                0 // retSize: `0` call via assembly to avoid memcopying a very large returndata
            )

            returnDataSize := returndatasize()
        }

        if (returnDataSize > returnDataSizeLimit) {
            returnDataSize = returnDataSizeLimit;
        }

        bytes memory returnData = new bytes(returnDataSize);
        assembly {
            returndatacopy(add(returnData, 0x20), 0, returnDataSize)
        }

        return (success, returnData);
    }
}