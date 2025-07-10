// SPDX-License-Identifier: MIT

// File @openzeppelin/contracts/utils/math/SafeMath.sol@v4.9.3

// Original license: SPDX_License_Identifier: MIT
// OpenZeppelin Contracts (last updated v4.9.0) (utils/math/SafeMath.sol)

pragma solidity ^0.8.0;

// CAUTION
// This version of SafeMath should only be used with Solidity 0.8 or later,
// because it relies on the compiler's built in overflow checks.

/**
 * @dev Wrappers over Solidity's arithmetic operations.
 *
 * NOTE: `SafeMath` is generally not needed starting with Solidity 0.8, since the compiler
 * now has built in overflow checking.
 */
library SafeMath {
    /**
     * @dev Returns the addition of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an overflow flag.
     *
     * _Available since v3.4._
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a division by zero flag.
     *
     * _Available since v3.4._
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a division by zero flag.
     *
     * _Available since v3.4._
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Returns the addition of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `+` operator.
     *
     * Requirements:
     *
     * - Addition cannot overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting on
     * overflow (when the result is negative).
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return a - b;
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, reverting on
     * overflow.
     *
     * Counterpart to Solidity's `*` operator.
     *
     * Requirements:
     *
     * - Multiplication cannot overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        return a * b;
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator.
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return a / b;
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting when dividing by zero.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return a % b;
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, reverting with custom message on
     * overflow (when the result is negative).
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {trySub}.
     *
     * Counterpart to Solidity's `-` operator.
     *
     * Requirements:
     *
     * - Subtraction cannot overflow.
     */
    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b <= a, errorMessage);
            return a - b;
        }
    }

    /**
     * @dev Returns the integer division of two unsigned integers, reverting with custom message on
     * division by zero. The result is rounded towards zero.
     *
     * Counterpart to Solidity's `/` operator. Note: this function uses a
     * `revert` opcode (which leaves remaining gas untouched) while Solidity
     * uses an invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a / b;
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers. (unsigned integer modulo),
     * reverting with custom message when dividing by zero.
     *
     * CAUTION: This function is deprecated because it requires allocating memory for the error
     * message unnecessarily. For custom revert reasons use {tryMod}.
     *
     * Counterpart to Solidity's `%` operator. This function uses a `revert`
     * opcode (which leaves remaining gas untouched) while Solidity uses an
     * invalid opcode to revert (consuming all remaining gas).
     *
     * Requirements:
     *
     * - The divisor cannot be zero.
     */
    function mod(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        unchecked {
            require(b > 0, errorMessage);
            return a % b;
        }
    }
}


// File contracts/crossApproach/lib/HTLCTxLib.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

library HTLCTxLib {
    using SafeMath for uint;

    /**
     *
     * ENUMS
     *
     */

    /// @notice tx info status
    /// @notice uninitialized,locked,redeemed,revoked
    enum TxStatus {None, Locked, Redeemed, Revoked, AssetLocked, DebtLocked}

    /**
     *
     * STRUCTURES
     *
     */

    /// @notice struct of HTLC user mint lock parameters
    struct HTLCUserParams {
        bytes32 xHash;                  /// hash of HTLC random number
        bytes32 smgID;                  /// ID of storeman group which user has selected
        uint tokenPairID;               /// token pair id on cross chain
        uint value;                     /// exchange token value
        uint lockFee;                   /// exchange token value
        uint lockedTime;                /// HTLC lock time
    }

    /// @notice HTLC(Hashed TimeLock Contract) tx info
    struct BaseTx {
        bytes32 smgID;                  /// HTLC transaction storeman ID
        uint lockedTime;                /// HTLC transaction locked time
        uint beginLockedTime;           /// HTLC transaction begin locked time
        TxStatus status;                /// HTLC transaction status
    }

    /// @notice user  tx info
    struct UserTx {
        BaseTx baseTx;
        uint tokenPairID;
        uint value;
        uint fee;
        address userAccount;            /// HTLC transaction sender address for the security check while user's revoke
    }
    /// @notice storeman  tx info
    struct SmgTx {
        BaseTx baseTx;
        uint tokenPairID;
        uint value;
        address  userAccount;          /// HTLC transaction user address for the security check while user's redeem
    }
    /// @notice storeman  debt tx info
    struct DebtTx {
        BaseTx baseTx;
        bytes32 srcSmgID;              /// HTLC transaction sender(source storeman) ID
    }

    struct Data {
        /// @notice mapping of hash(x) to UserTx -- xHash->htlcUserTxData
        mapping(bytes32 => UserTx) mapHashXUserTxs;

        /// @notice mapping of hash(x) to SmgTx -- xHash->htlcSmgTxData
        mapping(bytes32 => SmgTx) mapHashXSmgTxs;

        /// @notice mapping of hash(x) to DebtTx -- xHash->htlcDebtTxData
        mapping(bytes32 => DebtTx) mapHashXDebtTxs;

    }

    /**
     *
     * MANIPULATIONS
     *
     */

    /// @notice                     add user transaction info
    /// @param params               parameters for user tx
    function addUserTx(Data storage self, HTLCUserParams memory params)
        public
    {
        UserTx memory userTx = self.mapHashXUserTxs[params.xHash];
        // UserTx storage userTx = self.mapHashXUserTxs[params.xHash];
        // require(params.value != 0, "Value is invalid");
        require(userTx.baseTx.status == TxStatus.None, "User tx exists");

        userTx.baseTx.smgID = params.smgID;
        userTx.baseTx.lockedTime = params.lockedTime;
        userTx.baseTx.beginLockedTime = block.timestamp;
        userTx.baseTx.status = TxStatus.Locked;
        userTx.tokenPairID = params.tokenPairID;
        userTx.value = params.value;
        userTx.fee = params.lockFee;
        userTx.userAccount = msg.sender;

        self.mapHashXUserTxs[params.xHash] = userTx;
    }

    /// @notice                     refund coins from HTLC transaction, which is used for storeman redeem(outbound)
    /// @param x                    HTLC random number
    function redeemUserTx(Data storage self, bytes32 x)
        external
        returns(bytes32 xHash)
    {
        xHash = sha256(abi.encodePacked(x));

        UserTx storage userTx = self.mapHashXUserTxs[xHash];
        require(userTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(block.timestamp < userTx.baseTx.beginLockedTime.add(userTx.baseTx.lockedTime), "Redeem timeout");

        userTx.baseTx.status = TxStatus.Redeemed;

        return xHash;
    }

    /// @notice                     revoke user transaction
    /// @param  xHash               hash of HTLC random number
    function revokeUserTx(Data storage self, bytes32 xHash)
        external
    {
        UserTx storage userTx = self.mapHashXUserTxs[xHash];
        require(userTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(block.timestamp >= userTx.baseTx.beginLockedTime.add(userTx.baseTx.lockedTime), "Revoke is not permitted");

        userTx.baseTx.status = TxStatus.Revoked;
    }

    /// @notice                    function for get user info
    /// @param xHash               hash of HTLC random number
    /// @return smgID              ID of storeman which user has selected
    /// @return tokenPairID        token pair ID of cross chain
    /// @return value              exchange value
    /// @return fee                exchange fee
    /// @return userAccount        HTLC transaction sender address for the security check while user's revoke
    function getUserTx(Data storage self, bytes32 xHash)
        external
        view
        returns (bytes32, uint, uint, uint, address)
    {
        UserTx storage userTx = self.mapHashXUserTxs[xHash];
        return (userTx.baseTx.smgID, userTx.tokenPairID, userTx.value, userTx.fee, userTx.userAccount);
    }

    /// @notice                     add storeman transaction info
    /// @param  xHash               hash of HTLC random number
    /// @param  smgID               ID of the storeman which user has selected
    /// @param  tokenPairID         token pair ID of cross chain
    /// @param  value               HTLC transfer value of token
    /// @param  userAccount            user account address on the destination chain, which is used to redeem token
    function addSmgTx(Data storage self, bytes32 xHash, bytes32 smgID, uint tokenPairID, uint value, address userAccount, uint lockedTime)
        external
    {
        SmgTx memory smgTx = self.mapHashXSmgTxs[xHash];
        // SmgTx storage smgTx = self.mapHashXSmgTxs[xHash];
        require(value != 0, "Value is invalid");
        require(smgTx.baseTx.status == TxStatus.None, "Smg tx exists");

        smgTx.baseTx.smgID = smgID;
        smgTx.baseTx.status = TxStatus.Locked;
        smgTx.baseTx.lockedTime = lockedTime;
        smgTx.baseTx.beginLockedTime = block.timestamp;
        smgTx.tokenPairID = tokenPairID;
        smgTx.value = value;
        smgTx.userAccount = userAccount;

        self.mapHashXSmgTxs[xHash] = smgTx;
    }

    /// @notice                     refund coins from HTLC transaction, which is used for users redeem(inbound)
    /// @param x                    HTLC random number
    function redeemSmgTx(Data storage self, bytes32 x)
        external
        returns(bytes32 xHash)
    {
        xHash = sha256(abi.encodePacked(x));

        SmgTx storage smgTx = self.mapHashXSmgTxs[xHash];
        require(smgTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(block.timestamp < smgTx.baseTx.beginLockedTime.add(smgTx.baseTx.lockedTime), "Redeem timeout");

        smgTx.baseTx.status = TxStatus.Redeemed;

        return xHash;
    }

    /// @notice                     revoke storeman transaction
    /// @param  xHash               hash of HTLC random number
    function revokeSmgTx(Data storage self, bytes32 xHash)
        external
    {
        SmgTx storage smgTx = self.mapHashXSmgTxs[xHash];
        require(smgTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(block.timestamp >= smgTx.baseTx.beginLockedTime.add(smgTx.baseTx.lockedTime), "Revoke is not permitted");

        smgTx.baseTx.status = TxStatus.Revoked;
    }

    /// @notice                     function for get smg info
    /// @param xHash                hash of HTLC random number
    /// @return smgID               ID of storeman which user has selected
    /// @return tokenPairID         token pair ID of cross chain
    /// @return value               exchange value
    /// @return userAccount            user account address for redeem
    function getSmgTx(Data storage self, bytes32 xHash)
        external
        view
        returns (bytes32, uint, uint, address)
    {
        SmgTx storage smgTx = self.mapHashXSmgTxs[xHash];
        return (smgTx.baseTx.smgID, smgTx.tokenPairID, smgTx.value, smgTx.userAccount);
    }

    /// @notice                     add storeman transaction info
    /// @param  xHash               hash of HTLC random number
    /// @param  srcSmgID            ID of source storeman group
    /// @param  destSmgID           ID of the storeman which will take over of the debt of source storeman group
    /// @param  lockedTime          HTLC lock time
    /// @param  status              Status, should be 'Locked' for asset or 'DebtLocked' for debt
    function addDebtTx(Data storage self, bytes32 xHash, bytes32 srcSmgID, bytes32 destSmgID, uint lockedTime, TxStatus status)
        external
    {
        DebtTx memory debtTx = self.mapHashXDebtTxs[xHash];
        // DebtTx storage debtTx = self.mapHashXDebtTxs[xHash];
        require(debtTx.baseTx.status == TxStatus.None, "Debt tx exists");

        debtTx.baseTx.smgID = destSmgID;
        debtTx.baseTx.status = status;//TxStatus.Locked;
        debtTx.baseTx.lockedTime = lockedTime;
        debtTx.baseTx.beginLockedTime = block.timestamp;
        debtTx.srcSmgID = srcSmgID;

        self.mapHashXDebtTxs[xHash] = debtTx;
    }

    /// @notice                     refund coins from HTLC transaction
    /// @param x                    HTLC random number
    /// @param status               Status, should be 'Locked' for asset or 'DebtLocked' for debt
    function redeemDebtTx(Data storage self, bytes32 x, TxStatus status)
        external
        returns(bytes32 xHash)
    {
        xHash = sha256(abi.encodePacked(x));

        DebtTx storage debtTx = self.mapHashXDebtTxs[xHash];
        // require(debtTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(debtTx.baseTx.status == status, "Status is not locked");
        require(block.timestamp < debtTx.baseTx.beginLockedTime.add(debtTx.baseTx.lockedTime), "Redeem timeout");

        debtTx.baseTx.status = TxStatus.Redeemed;

        return xHash;
    }

    /// @notice                     revoke debt transaction, which is used for source storeman group
    /// @param  xHash               hash of HTLC random number
    /// @param  status              Status, should be 'Locked' for asset or 'DebtLocked' for debt
    function revokeDebtTx(Data storage self, bytes32 xHash, TxStatus status)
        external
    {
        DebtTx storage debtTx = self.mapHashXDebtTxs[xHash];
        // require(debtTx.baseTx.status == TxStatus.Locked, "Status is not locked");
        require(debtTx.baseTx.status == status, "Status is not locked");
        require(block.timestamp >= debtTx.baseTx.beginLockedTime.add(debtTx.baseTx.lockedTime), "Revoke is not permitted");

        debtTx.baseTx.status = TxStatus.Revoked;
    }

    /// @notice                     function for get debt info
    /// @param xHash                hash of HTLC random number
    /// @return srcSmgID            ID of source storeman
    /// @return destSmgID           ID of destination storeman
    function getDebtTx(Data storage self, bytes32 xHash)
        external
        view
        returns (bytes32, bytes32)
    {
        DebtTx storage debtTx = self.mapHashXDebtTxs[xHash];
        return (debtTx.srcSmgID, debtTx.baseTx.smgID);
    }

    function getLeftTime(uint endTime) private view returns (uint) {
        if (block.timestamp < endTime) {
            return endTime.sub(block.timestamp);
        }
        return 0;
    }

    /// @notice                     function for get debt info
    /// @param xHash                hash of HTLC random number
    /// @return leftTime            the left lock time
    function getLeftLockedTime(Data storage self, bytes32 xHash)
        external
        view
        returns (uint leftTime)
    {
        UserTx storage userTx = self.mapHashXUserTxs[xHash];
        if (userTx.baseTx.status != TxStatus.None) {
            return getLeftTime(userTx.baseTx.beginLockedTime.add(userTx.baseTx.lockedTime));
        }
        SmgTx storage smgTx = self.mapHashXSmgTxs[xHash];
        if (smgTx.baseTx.status != TxStatus.None) {
            return getLeftTime(smgTx.baseTx.beginLockedTime.add(smgTx.baseTx.lockedTime));
        }
        DebtTx storage debtTx = self.mapHashXDebtTxs[xHash];
        if (debtTx.baseTx.status != TxStatus.None) {
            return getLeftTime(debtTx.baseTx.beginLockedTime.add(debtTx.baseTx.lockedTime));
        }
        require(false, 'invalid xHash');
    }
}


// File contracts/crossApproach/lib/RapidityTxLib.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

library RapidityTxLib {

    /**
     *
     * ENUMS
     *
     */

    /// @notice tx info status
    /// @notice uninitialized,Redeemed
    enum TxStatus {None, Redeemed}

    /**
     *
     * STRUCTURES
     *
     */
    struct Data {
        /// @notice mapping of uniqueID to TxStatus -- uniqueID->TxStatus
        mapping(bytes32 => TxStatus) mapTxStatus;

    }

    /**
     *
     * MANIPULATIONS
     *
     */

    /// @notice                     add user transaction info
    /// @param  uniqueID            Rapidity random number
    function addRapidityTx(Data storage self, bytes32 uniqueID)
        internal
    {
        TxStatus status = self.mapTxStatus[uniqueID];
        require(status == TxStatus.None, "Rapidity tx exists");
        self.mapTxStatus[uniqueID] = TxStatus.Redeemed;
    }
}


// File contracts/interfaces/IQuota.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

interface IQuota {
  function userLock(uint tokenId, bytes32 storemanGroupId, uint value) external;
  function userBurn(uint tokenId, bytes32 storemanGroupId, uint value) external;

  function smgRelease(uint tokenId, bytes32 storemanGroupId, uint value) external;
  function smgMint(uint tokenId, bytes32 storemanGroupId, uint value) external;

  function upgrade(bytes32 storemanGroupId) external;

  function transferAsset(bytes32 srcStoremanGroupId, bytes32 dstStoremanGroupId) external;
  function receiveDebt(bytes32 srcStoremanGroupId, bytes32 dstStoremanGroupId) external;

  function getUserMintQuota(uint tokenId, bytes32 storemanGroupId) external view returns (uint);
  function getSmgMintQuota(uint tokenId, bytes32 storemanGroupId) external view returns (uint);

  function getUserBurnQuota(uint tokenId, bytes32 storemanGroupId) external view returns (uint);
  function getSmgBurnQuota(uint tokenId, bytes32 storemanGroupId) external view returns (uint);

  function getAsset(uint tokenId, bytes32 storemanGroupId) external view returns (uint asset, uint asset_receivable, uint asset_payable);
  function getDebt(uint tokenId, bytes32 storemanGroupId) external view returns (uint debt, uint debt_receivable, uint debt_payable);

  function isDebtClean(bytes32 storemanGroupId) external view returns (bool);
}


// File contracts/interfaces/IRC20Protocol.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

interface IRC20Protocol {
    function transfer(address, uint) external returns (bool);
    function transferFrom(address, address, uint) external returns (bool);
    function balanceOf(address _owner) external view returns (uint);
}


// File contracts/interfaces/ISignatureVerifier.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

interface ISignatureVerifier {
  function verify(
        uint curveId,
        bytes32 signature,
        bytes32 groupKeyX,
        bytes32 groupKeyY,
        bytes32 randomPointX,
        bytes32 randomPointY,
        bytes32 message
    ) external returns (bool);
}


// File contracts/interfaces/IStoremanGroup.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

interface IStoremanGroup {
    function getSelectedSmNumber(bytes32 groupId) external view returns(uint number);
    function getStoremanGroupConfig(bytes32 id) external view returns(bytes32 groupId, uint8 status, uint deposit, uint chain1, uint chain2, uint curve1, uint curve2,  bytes memory gpk1, bytes memory gpk2, uint startTime, uint endTime);
    function getDeposit(bytes32 id) external view returns(uint);
    function getStoremanGroupStatus(bytes32 id) external view returns(uint8 status, uint startTime, uint endTime);
    function setGpk(bytes32 groupId, bytes memory gpk1, bytes memory gpk2) external;
    function setInvalidSm(bytes32 groupId, uint[] memory indexs, uint8[] memory slashTypes) external returns(bool isContinue);
    function getThresholdByGrpId(bytes32 groupId) external view returns (uint);
    function getSelectedSmInfo(bytes32 groupId, uint index) external view returns(address wkAddr, bytes memory PK, bytes memory enodeId);
    function recordSmSlash(address wk) external;
}


// File contracts/interfaces/ITokenManager.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;

interface ITokenManager {
    function getTokenPairInfo(uint id) external view
      returns (uint origChainID, bytes memory tokenOrigAccount, uint shadowChainID, bytes memory tokenShadowAccount);

    function getTokenPairInfoSlim(uint id) external view
      returns (uint origChainID, bytes memory tokenOrigAccount, uint shadowChainID);

    function getAncestorInfo(uint id) external view
      returns (bytes memory account, string memory name, string memory symbol, uint8 decimals, uint chainId);

    function mintToken(address tokenAddress, address to, uint value) external;

    function burnToken(address tokenAddress, address from, uint value) external;

    function mapTokenPairType(uint tokenPairID) external view
      returns (uint8 tokenPairType);

    // erc1155
    function mintNFT(uint tokenCrossType, address tokenAddress, address to, uint[] calldata ids, uint[] calldata values, bytes calldata data) external;
    function burnNFT(uint tokenCrossType, address tokenAddress, address from, uint[] calldata ids, uint[] calldata values) external;
}


// File contracts/crossApproach/lib/CrossTypes.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;







library CrossTypes {
    using SafeMath for uint;

    /**
     *
     * STRUCTURES
     *
     */

    struct Data {

        /// map of the htlc transaction info
        HTLCTxLib.Data htlcTxData;

        /// map of the rapidity transaction info
        RapidityTxLib.Data rapidityTxData;

        /// quota data of storeman group
        IQuota quota;

        /// token manager instance interface
        ITokenManager tokenManager;

        /// storemanGroup admin instance interface
        IStoremanGroup smgAdminProxy;

        /// storemanGroup fee admin instance address
        address smgFeeProxy;

        ISignatureVerifier sigVerifier;

        /// @notice transaction fee, smgID => fee
        mapping(bytes32 => uint) mapStoremanFee;

        /// @notice transaction fee, origChainID => shadowChainID => fee
        mapping(uint => mapping(uint =>uint)) mapContractFee;

        /// @notice transaction fee, origChainID => shadowChainID => fee
        mapping(uint => mapping(uint =>uint)) mapAgentFee;

    }

    /**
     *
     * MANIPULATIONS
     *
     */

    /// @notice       convert bytes to address
    /// @param b      bytes
    function bytesToAddress(bytes memory b) internal pure returns (address addr) {
        assembly {
            addr := mload(add(b,20))
        }
    }

    function transfer(address tokenScAddr, address to, uint value)
        internal
        returns(bool)
    {
        uint beforeBalance;
        uint afterBalance;
        IRC20Protocol token = IRC20Protocol(tokenScAddr);
        beforeBalance = token.balanceOf(to);
        (bool success,) = tokenScAddr.call(abi.encodeWithSelector(token.transfer.selector, to, value));
        require(success, "transfer failed");
        afterBalance = token.balanceOf(to);
        return afterBalance == beforeBalance.add(value);
    }

    function transferFrom(address tokenScAddr, address from, address to, uint value)
        internal
        returns(bool)
    {
        uint beforeBalance;
        uint afterBalance;
        IRC20Protocol token = IRC20Protocol(tokenScAddr);
        beforeBalance = token.balanceOf(to);
        (bool success,) = tokenScAddr.call(abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
        require(success, "transferFrom failed");
        afterBalance = token.balanceOf(to);
        return afterBalance == beforeBalance.add(value);
    }
}


// File contracts/crossApproach/lib/EtherTransfer.sol

// Original license: SPDX_License_Identifier: MIT

pragma solidity ^0.8.1;

library EtherTransfer {
    /**
     * @dev Replacement for Solidity's `transfer`: sends `amount` wei to
     * `recipient`, forwarding all available gas and reverting on errors.
     *
     * https://eips.ethereum.org/EIPS/eip-1884[EIP1884] increases the gas cost
     * of certain opcodes, possibly making contracts go over the 2300 gas limit
     * imposed by `transfer`, making them unable to receive funds via
     * `transfer`. {sendValue} removes this limitation.
     *
     * https://consensys.net/diligence/blog/2023/09/stop-using-soliditys-transfer-now/[Learn more].
     *
     * IMPORTANT: because control is transferred to `recipient`, care must be
     * taken to not create reentrancy vulnerabilities. Consider using
     * {ReentrancyGuard} or the
     * https://solidity.readthedocs.io/en/v0.8.0/security-considerations.html#use-the-checks-effects-interactions-pattern[checks-effects-interactions pattern].
     */
    function sendValue(address payable recipient, uint256 amount, uint256 gasLimit) internal {
        require(address(this).balance >= amount, "EtherTransfer: insufficient balance");

        (bool success, ) = recipient.call{value: amount, gas: gasLimit}("");
        require(success, "EtherTransfer: unable to send value, recipient may have reverted");
    }
}


// File contracts/crossApproach/lib/RapidityLibV4.sol

// Original license: SPDX_License_Identifier: MIT

/*

  Copyright 2023 Wanchain Foundation.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

*/

//                            _           _           _
//  __      ____ _ _ __   ___| |__   __ _(_)_ __   __| | _____   __
//  \ \ /\ / / _` | '_ \ / __| '_ \ / _` | | '_ \@/ _` |/ _ \ \ / /
//   \ V  V / (_| | | | | (__| | | | (_| | | | | | (_| |  __/\ V /
//    \_/\_/ \__,_|_| |_|\___|_| |_|\__,_|_|_| |_|\__,_|\___| \_/
//
//

pragma solidity ^0.8.18;





library RapidityLibV4 {
    using SafeMath for uint;
    using RapidityTxLib for RapidityTxLib.Data;

    enum TokenCrossType {ERC20, ERC721, ERC1155}

    /**
    *
    * STRUCTURES
    *
    */

    /// @notice struct of Rapidity storeman mint lock parameters
    struct RapidityUserLockParams {
        bytes32 smgID;                    /// ID of storeman group which user has selected
        uint tokenPairID;                 /// token pair id on cross chain
        uint value;                       /// exchange token value
        uint currentChainID;              /// current chain ID
        uint tokenPairContractFee;        /// fee of token pair
        uint etherTransferGasLimit;     /// exchange token fee
        bytes destUserAccount;            /// account of shadow chain, used to receive token
        address smgFeeProxy;              /// address of the proxy to store fee for storeman group
    }

    /// @notice struct of Rapidity storeman mint lock parameters
    struct RapiditySmgMintParams {
        bytes32 uniqueID;                 /// Rapidity random number
        bytes32 smgID;                    /// ID of storeman group which user has selected
        uint tokenPairID;                 /// token pair id on cross chain
        uint value;                       /// exchange token value
        uint fee;                         /// exchange token fee
        address destTokenAccount;         /// shadow token account
        address destUserAccount;          /// account of shadow chain, used to receive token
        address smgFeeProxy;              /// address of the proxy to store fee for storeman group
    }

    /// @notice struct of Rapidity user burn lock parameters
    struct RapidityUserBurnParams {
        bytes32 smgID;                  /// ID of storeman group which user has selected
        uint tokenPairID;               /// token pair id on cross chain
        uint value;                     /// exchange token value
        uint currentChainID;            /// current chain ID
        uint fee;                       /// exchange token fee
        uint tokenPairContractFee;      /// fee of token pair
        uint etherTransferGasLimit;     /// exchange token fee
        address srcTokenAccount;        /// shadow token account
        bytes destUserAccount;          /// account of token destination chain, used to receive token
        address smgFeeProxy;            /// address of the proxy to store fee for storeman group
    }

    /// @notice struct of Rapidity user burn lock parameters
    struct RapiditySmgReleaseParams {
        bytes32 uniqueID;               /// Rapidity random number
        bytes32 smgID;                  /// ID of storeman group which user has selected
        uint tokenPairID;               /// token pair id on cross chain
        uint value;                     /// exchange token value
        uint fee;                       /// exchange token fee
        uint etherTransferGasLimit;     /// exchange token fee
        address destTokenAccount;       /// original token/coin account
        address destUserAccount;        /// account of token original chain, used to receive token
        address smgFeeProxy;            /// address of the proxy to store fee for storeman group
    }

    /**
     *
     * EVENTS
     *
     **/


    /// @notice                         event of exchange WRC-20 token with original chain token request
    /// @notice                         event invoked by storeman group
    /// @param smgID                    ID of storemanGroup
    /// @param tokenPairID              token pair ID of cross chain token
    /// @param tokenAccount             Rapidity original token account
    /// @param value                    Rapidity value
    /// @param userAccount              account of shadow chain, used to receive token
    event UserLockLogger(bytes32 indexed smgID, uint indexed tokenPairID, address indexed tokenAccount, uint value, uint contractFee, bytes userAccount);

    /// @notice                         event of exchange WRC-20 token with original chain token request
    /// @notice                         event invoked by storeman group
    /// @param smgID                    ID of storemanGroup
    /// @param tokenPairID              token pair ID of cross chain token
    /// @param tokenAccount             Rapidity shadow token account
    /// @param value                    Rapidity value
    /// @param userAccount              account of shadow chain, used to receive token
    event UserBurnLogger(bytes32 indexed smgID, uint indexed tokenPairID, address indexed tokenAccount, uint value, uint contractFee, uint fee, bytes userAccount);

    /// @notice                         event of exchange WRC-20 token with original chain token request
    /// @notice                         event invoked by storeman group
    /// @param uniqueID                 unique random number
    /// @param smgID                    ID of storemanGroup
    /// @param tokenPairID              token pair ID of cross chain token
    /// @param value                    Rapidity value
    /// @param tokenAccount             Rapidity shadow token account
    /// @param userAccount              account of original chain, used to receive token
    event SmgMintLogger(bytes32 indexed uniqueID, bytes32 indexed smgID, uint indexed tokenPairID, uint value, address tokenAccount, address userAccount);
    event SmgMint(bytes32 indexed uniqueID, bytes32 indexed smgID, uint indexed tokenPairID, string[] keys, bytes[] values);

    /// @notice                         event of exchange WRC-20 token with original chain token request
    /// @notice                         event invoked by storeman group
    /// @param uniqueID                 unique random number
    /// @param smgID                    ID of storemanGroup
    /// @param tokenPairID              token pair ID of cross chain token
    /// @param value                    Rapidity value
    /// @param tokenAccount             Rapidity original token account
    /// @param userAccount              account of original chain, used to receive token
    event SmgReleaseLogger(bytes32 indexed uniqueID, bytes32 indexed smgID, uint indexed tokenPairID, uint value, address tokenAccount, address userAccount);
    event SmgRelease(bytes32 indexed uniqueID, bytes32 indexed smgID, uint indexed tokenPairID, string[] keys, bytes[] values);

    /**
    *
    * MANIPULATIONS
    *
    */

    /// @notice                         mintBridge, user lock token on token original chain
    /// @notice                         event invoked by user mint lock
    /// @param storageData              Cross storage data
    /// @param params                   parameters for user mint lock token on token original chain
    function userLock(CrossTypes.Data storage storageData, RapidityUserLockParams memory params)
    public
    {
        ITokenManager tokenManager = storageData.tokenManager;
        uint fromChainID;
        uint toChainID;
        bytes memory fromTokenAccount;
        bytes memory toTokenAccount;
        (fromChainID,fromTokenAccount,toChainID,toTokenAccount) = tokenManager.getTokenPairInfo(params.tokenPairID);
        require(fromChainID != 0, "Token does not exist");

        uint contractFee = params.tokenPairContractFee;
        address tokenScAddr;
        if (params.currentChainID == fromChainID) {
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[fromChainID][toChainID];
            }
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[fromChainID][0];
            }
            tokenScAddr = CrossTypes.bytesToAddress(fromTokenAccount);
        } else if (params.currentChainID == toChainID) {
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[toChainID][fromChainID];
            }
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[toChainID][0];
            }
            tokenScAddr = CrossTypes.bytesToAddress(toTokenAccount);
        } else {
            require(false, "Invalid token pair");
        }
        if (contractFee > 0) {
            EtherTransfer.sendValue(payable(params.smgFeeProxy), contractFee, params.etherTransferGasLimit);
        }

        uint left;
        if (tokenScAddr == address(0)) {
            left = (msg.value).sub(params.value).sub(contractFee);
        } else {
            left = (msg.value).sub(contractFee);

            uint8 tokenCrossType = tokenManager.mapTokenPairType(params.tokenPairID);
            require(tokenCrossType == uint8(TokenCrossType.ERC20), "Not support");
            require(CrossTypes.transferFrom(tokenScAddr, msg.sender, address(this), params.value), "Lock token failed");
        }
        if (left != 0) {
            EtherTransfer.sendValue(payable(msg.sender), left, params.etherTransferGasLimit);
        }
        emit UserLockLogger(params.smgID, params.tokenPairID, tokenScAddr, params.value, contractFee, params.destUserAccount);
    }

    /// @notice                         burnBridge, user lock token on token original chain
    /// @notice                         event invoked by user burn lock
    /// @param storageData              Cross storage data
    /// @param params                   parameters for user burn lock token on token original chain
    function userBurn(CrossTypes.Data storage storageData, RapidityUserBurnParams memory params)
    public
    {
        ITokenManager tokenManager = storageData.tokenManager;
        uint fromChainID;
        uint toChainID;
        bytes memory fromTokenAccount;
        bytes memory toTokenAccount;
        (fromChainID,fromTokenAccount,toChainID,toTokenAccount) = tokenManager.getTokenPairInfo(params.tokenPairID);
        require(fromChainID != 0, "Token does not exist");

        uint256 contractFee = params.tokenPairContractFee;
        address tokenScAddr;
        if (params.currentChainID == toChainID) {
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[toChainID][fromChainID];
            }
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[toChainID][0];
            }
            tokenScAddr = CrossTypes.bytesToAddress(toTokenAccount);
        } else if (params.currentChainID == fromChainID) {
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[fromChainID][toChainID];
            }
            if (contractFee == 0) {
                contractFee = storageData.mapContractFee[fromChainID][0];
            }
            tokenScAddr = CrossTypes.bytesToAddress(fromTokenAccount);
        } else {
            require(false, "Invalid token pair");
        }
        require(params.srcTokenAccount == tokenScAddr, "Invalid token account");

        // Reuse variable fromChainID as tokenCrossType; burn token by tokenCrossType
        fromChainID = tokenManager.mapTokenPairType(params.tokenPairID);
        require(fromChainID == uint8(TokenCrossType.ERC20), "Not support");
        require(burnShadowToken(tokenManager, tokenScAddr, msg.sender, params.value), "Burn failed");

        if (contractFee > 0) {
            EtherTransfer.sendValue(payable(params.smgFeeProxy), contractFee, params.etherTransferGasLimit);
        }

        uint left = (msg.value).sub(contractFee);
        if (left != 0) {
            EtherTransfer.sendValue(payable(msg.sender), left, params.etherTransferGasLimit);
        }

        emit UserBurnLogger(params.smgID, params.tokenPairID, tokenScAddr, params.value, contractFee, params.fee, params.destUserAccount);
    }

    /// @notice                         mintBridge, storeman mint lock token on token shadow chain
    /// @notice                         event invoked by user mint lock
    /// @param storageData              Cross storage data
    /// @param params                   parameters for storeman mint lock token on token shadow chain
    function smgMint(CrossTypes.Data storage storageData, RapiditySmgMintParams memory params)
    public
    {
        storageData.rapidityTxData.addRapidityTx(params.uniqueID);

        ITokenManager tokenManager = storageData.tokenManager;
        uint8 tokenCrossType = tokenManager.mapTokenPairType(params.tokenPairID);
        require(tokenCrossType == uint8(TokenCrossType.ERC20), "Not support");
        if (params.fee > 0) {
            require(mintShadowToken(tokenManager, params.destTokenAccount, params.smgFeeProxy, params.fee), "Mint fee failed");
        }
        require(mintShadowToken(tokenManager, params.destTokenAccount, params.destUserAccount, params.value), "Mint failed");

        string[] memory keys = new string[](4);
        bytes[] memory values = new bytes[](4);
        keys[0] = "value:uint256";
        values[0] = abi.encodePacked(params.value);
        keys[1] = "tokenAccount:address";
        values[1] = abi.encodePacked(params.destTokenAccount);
        keys[2] = "userAccount:address";
        values[2] = abi.encodePacked(params.destUserAccount);
        keys[3] = "fee:uint256";
        values[3] = abi.encodePacked(params.fee);
        emit SmgMint(params.uniqueID, params.smgID, params.tokenPairID, keys, values);
        emit SmgMintLogger(params.uniqueID, params.smgID, params.tokenPairID, params.value, params.destTokenAccount, params.destUserAccount);
    }

    /// @notice                         burnBridge, storeman burn lock token on token shadow chain
    /// @notice                         event invoked by user burn lock
    /// @param storageData              Cross storage data
    /// @param params                   parameters for storeman burn lock token on token shadow chain
    function smgRelease(CrossTypes.Data storage storageData, RapiditySmgReleaseParams memory params)
    public
    {
        storageData.rapidityTxData.addRapidityTx(params.uniqueID);

        if (params.destTokenAccount == address(0)) {
            EtherTransfer.sendValue(payable(params.destUserAccount), params.value, params.etherTransferGasLimit);
            if (params.fee > 0) {
                EtherTransfer.sendValue(payable(params.smgFeeProxy), params.fee, params.etherTransferGasLimit);
            }
        } else {
            uint8 tokenCrossType = storageData.tokenManager.mapTokenPairType(params.tokenPairID);
            require(tokenCrossType == uint8(TokenCrossType.ERC20), "Not support");
            if (params.fee > 0) {
                require(CrossTypes.transfer(params.destTokenAccount, params.smgFeeProxy, params.fee), "Transfer token fee failed");
            }
            require(CrossTypes.transfer(params.destTokenAccount, params.destUserAccount, params.value), "Transfer token failed");
        }

        string[] memory keys = new string[](4);
        bytes[] memory values = new bytes[](4);
        keys[0] = "value:uint256";
        values[0] = abi.encodePacked(params.value);
        keys[1] = "tokenAccount:address";
        values[1] = abi.encodePacked(params.destTokenAccount);
        keys[2] = "userAccount:address";
        values[2] = abi.encodePacked(params.destUserAccount);
        keys[3] = "fee:uint256";
        values[3] = abi.encodePacked(params.fee);
        emit SmgRelease(params.uniqueID, params.smgID, params.tokenPairID, keys, values);
        emit SmgReleaseLogger(params.uniqueID, params.smgID, params.tokenPairID, params.value, params.destTokenAccount, params.destUserAccount);
    }

    function burnShadowToken(ITokenManager tokenManager, address tokenAddress, address userAccount, uint value) private returns (bool) {
        uint beforeBalance;
        uint afterBalance;
        beforeBalance = IRC20Protocol(tokenAddress).balanceOf(userAccount);

        tokenManager.burnToken(tokenAddress, userAccount, value);

        afterBalance = IRC20Protocol(tokenAddress).balanceOf(userAccount);
        return afterBalance == beforeBalance.sub(value);
    }

    function mintShadowToken(ITokenManager tokenManager, address tokenAddress, address userAccount, uint value) private returns (bool) {
        uint beforeBalance;
        uint afterBalance;
        beforeBalance = IRC20Protocol(tokenAddress).balanceOf(userAccount);

        tokenManager.mintToken(tokenAddress, userAccount, value);

        afterBalance = IRC20Protocol(tokenAddress).balanceOf(userAccount);
        return afterBalance == beforeBalance.add(value);
    }
}