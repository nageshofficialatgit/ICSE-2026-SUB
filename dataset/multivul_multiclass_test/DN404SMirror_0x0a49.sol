// SPDX-License-Identifier: UNLICENSED
// File: dn404/src/DN404Mirror.sol


pragma solidity ^0.8.4;

/// @title DN404sMirror
/// @notice DN404Mirror provides an interface for interacting with the
/// NFT tokens in a DN404 implementation.
///
/// ⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶ
/// Ⓐ    Trulady & Truman 404    Ⓐ
/// ⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶⒶ
///
/// @author  Ⓐ
///
///
/// @author  vectorized.eth (@optimizoor)
///          Thanks optimizoor
///
/// @dev Note:
/// - The ERC721 data is stored in the base DN404 contract.
contract DN404Mirror {
    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                           EVENTS                           */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Emitted when token `id` is transferred from `from` to `to`.
    event Transfer(address indexed from, address indexed to, uint256 indexed id);

    /// @dev Emitted when `owner` enables `account` to manage the `id` token.
    event Approval(address indexed owner, address indexed account, uint256 indexed id);

    /// @dev Emitted when `owner` enables or disables `operator` to manage all of their tokens.
    event ApprovalForAll(address indexed owner, address indexed operator, bool isApproved);

    /// @dev The ownership is transferred from `oldOwner` to `newOwner`.
    /// This is for marketplace signaling purposes. This contract has a `pullOwner()`
    /// function that will sync the owner from the base contract.
    event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

    /// @dev `keccak256(bytes("Transfer(address,address,uint256)"))`.
    uint256 private constant _TRANSFER_EVENT_SIGNATURE =
        0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef;

    /// @dev `keccak256(bytes("Approval(address,address,uint256)"))`.
    uint256 private constant _APPROVAL_EVENT_SIGNATURE =
        0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925;

    /// @dev `keccak256(bytes("ApprovalForAll(address,address,bool)"))`.
    uint256 private constant _APPROVAL_FOR_ALL_EVENT_SIGNATURE =
        0x17307eab39ab6107e8899845ad3d59bd9653f200f220920489ca2b5937696c31;

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                        CUSTOM ERRORS                       */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Thrown when a call for an NFT function did not originate
    /// from the base DN404 contract.
    error SenderNotBase();

    /// @dev Thrown when a call for an NFT function did not originate from the deployer.
    error SenderNotDeployer();

    /// @dev Thrown when transferring an NFT to a contract address that
    /// does not implement ERC721Receiver.
    error TransferToNonERC721ReceiverImplementer();

    /// @dev Thrown when a linkMirrorContract call is received and the
    /// NFT mirror contract has already been linked to a DN404 base contract.
    error AlreadyLinked();

    /// @dev Thrown when retrieving the base DN404 address when a link has not
    /// been established.
    error NotLinked();

    /// @dev The function selector is not recognized.
    error FnSelectorNotRecognized();

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                          STORAGE                           */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Struct contain the NFT mirror contract storage.
    struct DN404NFTStorage {
        // Address of the ERC20 base contract.
        address baseERC20;
        // The deployer, if provided. If non-zero, the initialization of the
        // ERC20 <-> ERC721 link can only be done by the deployer via the ERC20 base contract.
        address deployer;
        // The owner of the ERC20 base contract. For marketplace signaling.
        address owner;
    }

    /// @dev Returns a storage pointer for DN404NFTStorage.
    function _getDN404NFTStorage() internal pure virtual returns (DN404NFTStorage storage $) {
        /// @solidity memory-safe-assembly
        assembly {
            // `uint72(bytes9(keccak256("DN404_MIRROR_STORAGE")))`.
            $.slot := 0x3602298b8c10b01230 // Truncate to 9 bytes to reduce bytecode size.
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                        CONSTRUCTOR                         */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    constructor(address deployer) {
        // For non-proxies, we will store the deployer so that only the deployer can
        // link the base contract.
        _getDN404NFTStorage().deployer = deployer;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                     ERC721 OPERATIONS                      */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the token collection name from the base DN404 contract.
    function name() public view virtual returns (string memory) {
        return _readString(0x06fdde03, 0); // `name()`.
    }

    /// @dev Returns the token collection symbol from the base DN404 contract.
    function symbol() public view virtual returns (string memory) {
        return _readString(0x95d89b41, 0); // `symbol()`.
    }

    /// @dev Returns the Uniform Resource Identifier (URI) for token `id` from
    /// the base DN404 contract.
    function tokenURI(uint256 id) public view virtual returns (string memory) {
        ownerOf(id); // `ownerOf` reverts if the token does not exist.
        // We'll leave if optional for `_tokenURI` to revert for non-existent token
        // on the ERC20 side, since this is only recommended by the ERC721 standard.
        return _readString(0xcb30b460, id); // `tokenURINFT(uint256)`.
    }

    /// @dev Returns the total NFT supply from the base DN404 contract.
    function totalSupply() public view virtual returns (uint256) {
        return _readWord(0xe2c79281, 0, 0); // `totalNFTSupply()`.
    }

    /// @dev Returns the number of NFT tokens owned by `nftOwner` from the base DN404 contract.
    ///
    /// Requirements:
    /// - `nftOwner` must not be the zero address.
    function balanceOf(address nftOwner) public view virtual returns (uint256) {
        return _readWord(0xf5b100ea, uint160(nftOwner), 0); // `balanceOfNFT(address)`.
    }

    /// @dev Returns the owner of token `id` from the base DN404 contract.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function ownerOf(uint256 id) public view virtual returns (address) {
        return address(uint160(_readWord(0x2d8a746e, id, 0))); // `ownerOfNFT(uint256)`.
    }

    /// @dev Returns the owner of token `id` from the base DN404 contract.
    /// Returns `address(0)` instead of reverting if the token does not exist.
    function ownerAt(uint256 id) public view virtual returns (address) {
        return address(uint160(_readWord(0xc016aa52, id, 0))); // `ownerAtNFT(uint256)`.
    }

    /// @dev Sets `spender` as the approved account to manage token `id` in
    /// the base DN404 contract.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    /// - The caller must be the owner of the token,
    ///   or an approved operator for the token owner.
    ///
    /// Emits an {Approval} event.
    function approve(address spender, uint256 id) public payable virtual {
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            spender := shr(96, shl(96, spender))
            let m := mload(0x40)
            mstore(0x00, 0xd10b6e0c) // `approveNFT(address,uint256,address)`.
            mstore(0x20, spender)
            mstore(0x40, id)
            mstore(0x60, caller())
            if iszero(
                and( // Arguments of `and` are evaluated last to first.
                    gt(returndatasize(), 0x1f), // The call must return at least 32 bytes.
                    call(gas(), base, callvalue(), 0x1c, 0x64, 0x00, 0x20)
                )
            ) {
                returndatacopy(m, 0x00, returndatasize())
                revert(m, returndatasize())
            }
            mstore(0x40, m) // Restore the free memory pointer.
            mstore(0x60, 0) // Restore the zero pointer.
            // Emit the {Approval} event.
            log4(codesize(), 0x00, _APPROVAL_EVENT_SIGNATURE, shr(96, mload(0x0c)), spender, id)
        }
    }

    /// @dev Returns the account approved to manage token `id` from
    /// the base DN404 contract.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function getApproved(uint256 id) public view virtual returns (address) {
        return address(uint160(_readWord(0x27ef5495, id, 0))); // `getApprovedNFT(uint256)`.
    }

    /// @dev Sets whether `operator` is approved to manage the tokens of the caller in
    /// the base DN404 contract.
    ///
    /// Emits an {ApprovalForAll} event.
    function setApprovalForAll(address operator, bool approved) public virtual {
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            operator := shr(96, shl(96, operator))
            let m := mload(0x40)
            mstore(0x00, 0xf6916ddd) // `setApprovalForAllNFT(address,bool,address)`.
            mstore(0x20, operator)
            mstore(0x40, iszero(iszero(approved)))
            mstore(0x60, caller())
            if iszero(
                and( // Arguments of `and` are evaluated last to first.
                    eq(mload(0x00), 1), // The call must return 1.
                    call(gas(), base, callvalue(), 0x1c, 0x64, 0x00, 0x20)
                )
            ) {
                returndatacopy(m, 0x00, returndatasize())
                revert(m, returndatasize())
            }
            // Emit the {ApprovalForAll} event.
            // The `approved` value is already at 0x40.
            log3(0x40, 0x20, _APPROVAL_FOR_ALL_EVENT_SIGNATURE, caller(), operator)
            mstore(0x40, m) // Restore the free memory pointer.
            mstore(0x60, 0) // Restore the zero pointer.
        }
    }

    /// @dev Returns whether `operator` is approved to manage the tokens of `nftOwner` from
    /// the base DN404 contract.
    function isApprovedForAll(address nftOwner, address operator)
        public
        view
        virtual
        returns (bool)
    {
        // `isApprovedForAllNFT(address,address)`.
        return _readWord(0x62fb246d, uint160(nftOwner), uint160(operator)) != 0;
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - The caller must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function transferFrom(address from, address to, uint256 id) public payable virtual {
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            from := shr(96, shl(96, from))
            to := shr(96, shl(96, to))
            let m := mload(0x40)
            mstore(m, 0xe5eb36c8) // `transferFromNFT(address,address,uint256,address)`.
            mstore(add(m, 0x20), from)
            mstore(add(m, 0x40), to)
            mstore(add(m, 0x60), id)
            mstore(add(m, 0x80), caller())
            if iszero(
                and( // Arguments of `and` are evaluated last to first.
                    eq(mload(m), 1), // The call must return 1.
                    call(gas(), base, callvalue(), add(m, 0x1c), 0x84, m, 0x20)
                )
            ) {
                returndatacopy(m, 0x00, returndatasize())
                revert(m, returndatasize())
            }
            // Emit the {Transfer} event.
            log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, from, to, id)
        }
    }

    /// @dev Equivalent to `safeTransferFrom(from, to, id, "")`.
    function safeTransferFrom(address from, address to, uint256 id) public payable virtual {
        transferFrom(from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, "");
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - The caller must be the owner of the token, or be approved to manage the token.
    /// - If `to` refers to a smart contract, it must implement
    ///   {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
    ///
    /// Emits a {Transfer} event.
    function safeTransferFrom(address from, address to, uint256 id, bytes calldata data)
        public
        payable
        virtual
    {
        transferFrom(from, to, id);
        if (_hasCode(to)) _checkOnERC721Received(from, to, id, data);
    }

    /// @dev Returns true if this contract implements the interface defined by `interfaceId`.
    /// See: https://eips.ethereum.org/EIPS/eip-165
    /// This function call must use less than 30000 gas.
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            let s := shr(224, interfaceId)
            // ERC165: 0x01ffc9a7, ERC721: 0x80ac58cd, ERC721Metadata: 0x5b5e139f.
            result := or(or(eq(s, 0x01ffc9a7), eq(s, 0x80ac58cd)), eq(s, 0x5b5e139f))
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                  OWNER SYNCING OPERATIONS                  */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the `owner` of the contract, for marketplace signaling purposes.
    function owner() public view virtual returns (address) {
        return _getDN404NFTStorage().owner;
    }

    /// @dev Permissionless function to pull the owner from the base DN404 contract
    /// if it implements ownable, for marketplace signaling purposes.
    function pullOwner() public virtual returns (bool) {
        address newOwner;
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, 0x8da5cb5b) // `owner()`.
            let success := staticcall(gas(), base, 0x1c, 0x04, 0x00, 0x20)
            newOwner := mul(shr(96, mload(0x0c)), and(gt(returndatasize(), 0x1f), success))
        }
        DN404NFTStorage storage $ = _getDN404NFTStorage();
        address oldOwner = $.owner;
        if (oldOwner != newOwner) {
            $.owner = newOwner;
            emit OwnershipTransferred(oldOwner, newOwner);
        }
        return true;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                     MIRROR OPERATIONS                      */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the address of the base DN404 contract.
    function baseERC20() public view virtual returns (address base) {
        base = _getDN404NFTStorage().baseERC20;
        if (base == address(0)) revert NotLinked();
    }

    /// @dev Fallback modifier to execute calls from the base DN404 contract.
    modifier dn404NFTFallback() virtual {
        DN404NFTStorage storage $ = _getDN404NFTStorage();

        uint256 fnSelector = _calldataload(0x00) >> 224;

        // `logTransfer(uint256[])`.
        if (fnSelector == 0x263c69d6) {
            if (msg.sender != $.baseERC20) revert SenderNotBase();
            /// @solidity memory-safe-assembly
            assembly {
                let o := add(0x24, calldataload(0x04)) // Packed logs offset.
                let end := add(o, shl(5, calldataload(sub(o, 0x20))))
                for {} iszero(eq(o, end)) { o := add(0x20, o) } {
                    let d := calldataload(o) // Entry in the packed logs.
                    let a := shr(96, d) // The address.
                    let b := and(1, d) // Whether it is a burn.
                    log4(
                        codesize(),
                        0x00,
                        _TRANSFER_EVENT_SIGNATURE,
                        mul(a, b), // `from`.
                        mul(a, iszero(b)), // `to`.
                        shr(168, shl(160, d)) // `id`.
                    )
                }
                mstore(0x00, 0x01)
                return(0x00, 0x20)
            }
        }
        // `logDirectTransfer(address,address,uint256[])`.
        if (fnSelector == 0x144027d3) {
            if (msg.sender != $.baseERC20) revert SenderNotBase();
            /// @solidity memory-safe-assembly
            assembly {
                let from := calldataload(0x04)
                let to := calldataload(0x24)
                let o := add(0x24, calldataload(0x44)) // Direct logs offset.
                let end := add(o, shl(5, calldataload(sub(o, 0x20))))
                for {} iszero(eq(o, end)) { o := add(0x20, o) } {
                    log4(codesize(), 0x00, _TRANSFER_EVENT_SIGNATURE, from, to, calldataload(o))
                }
                mstore(0x00, 0x01)
                return(0x00, 0x20)
            }
        }
        // `linkMirrorContract(address)`.
        if (fnSelector == 0x0f4599e5) {
            if ($.deployer != address(0)) {
                if (address(uint160(_calldataload(0x04))) != $.deployer) {
                    revert SenderNotDeployer();
                }
            }
            if ($.baseERC20 != address(0)) revert AlreadyLinked();
            $.baseERC20 = msg.sender;
            /// @solidity memory-safe-assembly
            assembly {
                mstore(0x00, 0x01)
                return(0x00, 0x20)
            }
        }
        _;
    }

    /// @dev Fallback function for calls from base DN404 contract.
    /// Override this if you need to implement your custom
    /// fallback with utilities like Solady's `LibZip.cdFallback()`.
    /// And always remember to always wrap the fallback with `dn404NFTFallback`.
    fallback() external payable virtual dn404NFTFallback {
        revert FnSelectorNotRecognized(); // Not mandatory. Just for quality of life.
    }

    /// @dev This is to silence the compiler warning.
    /// Override and remove the revert if you want your contract to receive ETH via receive.
    receive() external payable virtual {
        if (msg.value != 0) revert();
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                      PRIVATE HELPERS                       */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Helper to read a string from the base DN404 contract.
    function _readString(uint256 fnSelector, uint256 arg0)
        private
        view
        returns (string memory result)
    {
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            mstore(0x00, fnSelector)
            mstore(0x20, arg0)
            if iszero(staticcall(gas(), base, 0x1c, 0x24, 0x00, 0x00)) {
                returndatacopy(result, 0x00, returndatasize())
                revert(result, returndatasize())
            }
            returndatacopy(0x00, 0x00, 0x20) // Copy the offset of the string in returndata.
            returndatacopy(result, mload(0x00), 0x20) // Copy the length of the string.
            returndatacopy(add(result, 0x20), add(mload(0x00), 0x20), mload(result)) // Copy the string.
            let end := add(add(result, 0x20), mload(result))
            mstore(end, 0) // Zeroize the word after the string.
            mstore(0x40, add(end, 0x20)) // Allocate memory.
        }
    }

    /// @dev Helper to read a word from the base DN404 contract.
    function _readWord(uint256 fnSelector, uint256 arg0, uint256 arg1)
        private
        view
        returns (uint256 result)
    {
        address base = baseERC20();
        /// @solidity memory-safe-assembly
        assembly {
            let m := mload(0x40)
            mstore(0x00, fnSelector)
            mstore(0x20, arg0)
            mstore(0x40, arg1)
            if iszero(
                and( // Arguments of `and` are evaluated last to first.
                    gt(returndatasize(), 0x1f), // The call must return at least 32 bytes.
                    staticcall(gas(), base, 0x1c, 0x44, 0x00, 0x20)
                )
            ) {
                returndatacopy(m, 0x00, returndatasize())
                revert(m, returndatasize())
            }
            mstore(0x40, m) // Restore the free memory pointer.
            result := mload(0x00)
        }
    }

    /// @dev Returns the calldata value at `offset`.
    function _calldataload(uint256 offset) private pure returns (uint256 value) {
        /// @solidity memory-safe-assembly
        assembly {
            value := calldataload(offset)
        }
    }

    /// @dev Returns if `a` has bytecode of non-zero length.
    function _hasCode(address a) private view returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := extcodesize(a) // Can handle dirty upper bits.
        }
    }

    /// @dev Perform a call to invoke {IERC721Receiver-onERC721Received} on `to`.
    /// Reverts if the target does not support the function correctly.
    function _checkOnERC721Received(address from, address to, uint256 id, bytes memory data)
        private
    {
        /// @solidity memory-safe-assembly
        assembly {
            // Prepare the calldata.
            let m := mload(0x40)
            let onERC721ReceivedSelector := 0x150b7a02
            mstore(m, onERC721ReceivedSelector)
            mstore(add(m, 0x20), caller()) // The `operator`, which is always `msg.sender`.
            mstore(add(m, 0x40), shr(96, shl(96, from)))
            mstore(add(m, 0x60), id)
            mstore(add(m, 0x80), 0x80)
            let n := mload(data)
            mstore(add(m, 0xa0), n)
            if n { pop(staticcall(gas(), 4, add(data, 0x20), n, add(m, 0xc0), n)) }
            // Revert if the call reverts.
            if iszero(call(gas(), to, 0, add(m, 0x1c), add(n, 0xa4), m, 0x20)) {
                if returndatasize() {
                    // Bubble up the revert if the call reverts.
                    returndatacopy(m, 0x00, returndatasize())
                    revert(m, returndatasize())
                }
            }
            // Load the returndata and compare it.
            if iszero(eq(mload(m), shl(224, onERC721ReceivedSelector))) {
                mstore(0x00, 0xd1a57ed6) // `TransferToNonERC721ReceiverImplementer()`.
                revert(0x1c, 0x04)
            }
        }
    }
}

// File: @openzeppelin/contracts/utils/Context.sol


// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/access/OwnablePermissions.sol


pragma solidity ^0.8.4;


abstract contract OwnablePermissions is Context {
    function _requireCallerIsContractOwner() internal view virtual;
}

// File: @openzeppelin/contracts/utils/introspection/IERC165.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[EIP].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
    /**
     * @dev Returns true if this contract implements the interface defined by
     * `interfaceId`. See the corresponding
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[EIP section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/interfaces/IEOARegistry.sol


pragma solidity ^0.8.4;


interface IEOARegistry is IERC165 {
    function isVerifiedEOA(address account) external view returns (bool);
}
// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/utils/TransferPolicy.sol


pragma solidity ^0.8.4;

enum AllowlistTypes {
    Operators,
    PermittedContractReceivers
}

enum ReceiverConstraints {
    None,
    NoCode,
    EOA
}

enum CallerConstraints {
    None,
    OperatorWhitelistEnableOTC,
    OperatorWhitelistDisableOTC
}

enum StakerConstraints {
    None,
    CallerIsTxOrigin,
    EOA
}

enum TransferSecurityLevels {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six
}

struct TransferSecurityPolicy {
    CallerConstraints callerConstraints;
    ReceiverConstraints receiverConstraints;
}

struct CollectionSecurityPolicy {
    TransferSecurityLevels transferSecurityLevel;
    uint120 operatorWhitelistId;
    uint120 permittedContractReceiversId;
}

// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/interfaces/ITransferSecurityRegistry.sol


pragma solidity ^0.8.4;


interface ITransferSecurityRegistry {
    event AddedToAllowlist(AllowlistTypes indexed kind, uint256 indexed id, address indexed account);
    event CreatedAllowlist(AllowlistTypes indexed kind, uint256 indexed id, string indexed name);
    event ReassignedAllowlistOwnership(AllowlistTypes indexed kind, uint256 indexed id, address indexed newOwner);
    event RemovedFromAllowlist(AllowlistTypes indexed kind, uint256 indexed id, address indexed account);
    event SetAllowlist(AllowlistTypes indexed kind, address indexed collection, uint120 indexed id);
    event SetTransferSecurityLevel(address indexed collection, TransferSecurityLevels level);

    function createOperatorWhitelist(string calldata name) external returns (uint120);
    function createPermittedContractReceiverAllowlist(string calldata name) external returns (uint120);
    function reassignOwnershipOfOperatorWhitelist(uint120 id, address newOwner) external;
    function reassignOwnershipOfPermittedContractReceiverAllowlist(uint120 id, address newOwner) external;
    function renounceOwnershipOfOperatorWhitelist(uint120 id) external;
    function renounceOwnershipOfPermittedContractReceiverAllowlist(uint120 id) external;
    function setTransferSecurityLevelOfCollection(address collection, TransferSecurityLevels level) external;
    function setOperatorWhitelistOfCollection(address collection, uint120 id) external;
    function setPermittedContractReceiverAllowlistOfCollection(address collection, uint120 id) external;
    function addOperatorToWhitelist(uint120 id, address operator) external;
    function addPermittedContractReceiverToAllowlist(uint120 id, address receiver) external;
    function removeOperatorFromWhitelist(uint120 id, address operator) external;
    function removePermittedContractReceiverFromAllowlist(uint120 id, address receiver) external;
    function getCollectionSecurityPolicy(address collection) external view returns (CollectionSecurityPolicy memory);
    function getWhitelistedOperators(uint120 id) external view returns (address[] memory);
    function getPermittedContractReceivers(uint120 id) external view returns (address[] memory);
    function isOperatorWhitelisted(uint120 id, address operator) external view returns (bool);
    function isContractReceiverPermitted(uint120 id, address receiver) external view returns (bool);
}
// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/interfaces/ITransferValidator.sol


pragma solidity ^0.8.4;


interface ITransferValidator {
    function applyCollectionTransferPolicy(address caller, address from, address to) external view;
}
// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/interfaces/ICreatorTokenTransferValidator.sol


pragma solidity ^0.8.4;




interface ICreatorTokenTransferValidator is ITransferSecurityRegistry, ITransferValidator, IEOARegistry {}
// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/interfaces/ICreatorToken.sol


pragma solidity ^0.8.4;


interface ICreatorToken {
    event TransferValidatorUpdated(address oldValidator, address newValidator);

    function getTransferValidator() external view returns (ICreatorTokenTransferValidator);
    function getSecurityPolicy() external view returns (CollectionSecurityPolicy memory);
    function getWhitelistedOperators() external view returns (address[] memory);
    function getPermittedContractReceivers() external view returns (address[] memory);
    function isOperatorWhitelisted(address operator) external view returns (bool);
    function isContractReceiverPermitted(address receiver) external view returns (bool);
    function isTransferAllowed(address caller, address from, address to) external view returns (bool);
}

// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/utils/TransferValidation.sol


pragma solidity ^0.8.4;


/**
 * @title TransferValidation
 * @author Limit Break, Inc.
 * @notice A mix-in that can be combined with ERC-721 contracts to provide more granular hooks.
 * Openzeppelin's ERC721 contract only provides hooks for before and after transfer.  This allows
 * developers to validate or customize transfers within the context of a mint, a burn, or a transfer.
 */
abstract contract TransferValidation is Context {
    
    error ShouldNotMintToBurnAddress();

    /// @dev Inheriting contracts should call this function in the _beforeTokenTransfer function to get more granular hooks.
    function _validateBeforeTransfer(address from, address to, uint256 tokenId) internal virtual {
        bool fromZeroAddress = from == address(0);
        bool toZeroAddress = to == address(0);

        if(fromZeroAddress && toZeroAddress) {
            revert ShouldNotMintToBurnAddress();
        } else if(fromZeroAddress) {
            _preValidateMint(_msgSender(), to, tokenId, msg.value);
        } else if(toZeroAddress) {
            _preValidateBurn(_msgSender(), from, tokenId, msg.value);
        } else {
            _preValidateTransfer(_msgSender(), from, to, tokenId, msg.value);
        }
    }

    /// @dev Inheriting contracts should call this function in the _afterTokenTransfer function to get more granular hooks.
    function _validateAfterTransfer(address from, address to, uint256 tokenId) internal virtual {
        bool fromZeroAddress = from == address(0);
        bool toZeroAddress = to == address(0);

        if(fromZeroAddress && toZeroAddress) {
            revert ShouldNotMintToBurnAddress();
        } else if(fromZeroAddress) {
            _postValidateMint(_msgSender(), to, tokenId, msg.value);
        } else if(toZeroAddress) {
            _postValidateBurn(_msgSender(), from, tokenId, msg.value);
        } else {
            _postValidateTransfer(_msgSender(), from, to, tokenId, msg.value);
        }
    }

    /// @dev Optional validation hook that fires before a mint
    function _preValidateMint(address caller, address to, uint256 tokenId, uint256 value) internal virtual {}

    /// @dev Optional validation hook that fires after a mint
    function _postValidateMint(address caller, address to, uint256 tokenId, uint256 value) internal virtual {}

    /// @dev Optional validation hook that fires before a burn
    function _preValidateBurn(address caller, address from, uint256 tokenId, uint256 value) internal virtual {}

    /// @dev Optional validation hook that fires after a burn
    function _postValidateBurn(address caller, address from, uint256 tokenId, uint256 value) internal virtual {}

    /// @dev Optional validation hook that fires before a transfer
    function _preValidateTransfer(address caller, address from, address to, uint256 tokenId, uint256 value) internal virtual {}

    /// @dev Optional validation hook that fires after a transfer
    function _postValidateTransfer(address caller, address from, address to, uint256 tokenId, uint256 value) internal virtual {}
}

// File: @openzeppelin/contracts/interfaces/IERC165.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC165.sol)

pragma solidity ^0.8.20;


// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/utils/CreatorTokenBase.sol


pragma solidity ^0.8.4;






/**
 * @title CreatorTokenBase
 * @author Limit Break, Inc.
 * @notice CreatorTokenBase is an abstract contract that provides basic functionality for managing token 
 * transfer policies through an implementation of ICreatorTokenTransferValidator. This contract is intended to be used
 * as a base for creator-specific token contracts, enabling customizable transfer restrictions and security policies.
 *
 * <h4>Features:</h4>
 * <ul>Ownable: This contract can have an owner who can set and update the transfer validator.</ul>
 * <ul>TransferValidation: Implements the basic token transfer validation interface.</ul>
 * <ul>ICreatorToken: Implements the interface for creator tokens, providing view functions for token security policies.</ul>
 *
 * <h4>Benefits:</h4>
 * <ul>Provides a flexible and modular way to implement custom token transfer restrictions and security policies.</ul>
 * <ul>Allows creators to enforce policies such as whitelisted operators and permitted contract receivers.</ul>
 * <ul>Can be easily integrated into other token contracts as a base contract.</ul>
 *
 * <h4>Intended Usage:</h4>
 * <ul>Use as a base contract for creator token implementations that require advanced transfer restrictions and 
 *   security policies.</ul>
 * <ul>Set and update the ICreatorTokenTransferValidator implementation contract to enforce desired policies for the 
 *   creator token.</ul>
 */
abstract contract CreatorTokenBase is OwnablePermissions, TransferValidation, ICreatorToken {
    
    error CreatorTokenBase__InvalidTransferValidatorContract();
    error CreatorTokenBase__SetTransferValidatorFirst();

    address public constant DEFAULT_TRANSFER_VALIDATOR = address(0x0000721C310194CcfC01E523fc93C9cCcFa2A0Ac);
    TransferSecurityLevels public constant DEFAULT_TRANSFER_SECURITY_LEVEL = TransferSecurityLevels.One;
    uint120 public constant DEFAULT_OPERATOR_WHITELIST_ID = uint120(1);

    ICreatorTokenTransferValidator private transferValidator;

    /**
     * @notice Allows the contract owner to set the transfer validator to the official validator contract
     *         and set the security policy to the recommended default settings.
     * @dev    May be overridden to change the default behavior of an individual collection.
     */
    function setToDefaultSecurityPolicy() public virtual {
        _requireCallerIsContractOwner();
        setTransferValidator(DEFAULT_TRANSFER_VALIDATOR);
        ICreatorTokenTransferValidator(DEFAULT_TRANSFER_VALIDATOR).setTransferSecurityLevelOfCollection(address(this), DEFAULT_TRANSFER_SECURITY_LEVEL);
        ICreatorTokenTransferValidator(DEFAULT_TRANSFER_VALIDATOR).setOperatorWhitelistOfCollection(address(this), DEFAULT_OPERATOR_WHITELIST_ID);
    }

    /**
     * @notice Allows the contract owner to set the transfer validator to a custom validator contract
     *         and set the security policy to their own custom settings.
     */
    function setToCustomValidatorAndSecurityPolicy(
        address validator, 
        TransferSecurityLevels level, 
        uint120 operatorWhitelistId, 
        uint120 permittedContractReceiversAllowlistId) public {
        _requireCallerIsContractOwner();

        setTransferValidator(validator);

        ICreatorTokenTransferValidator(validator).
            setTransferSecurityLevelOfCollection(address(this), level);

        ICreatorTokenTransferValidator(validator).
            setOperatorWhitelistOfCollection(address(this), operatorWhitelistId);

        ICreatorTokenTransferValidator(validator).
            setPermittedContractReceiverAllowlistOfCollection(address(this), permittedContractReceiversAllowlistId);
    }

    /**
     * @notice Allows the contract owner to set the security policy to their own custom settings.
     * @dev    Reverts if the transfer validator has not been set.
     */
    function setToCustomSecurityPolicy(
        TransferSecurityLevels level, 
        uint120 operatorWhitelistId, 
        uint120 permittedContractReceiversAllowlistId) public {
        _requireCallerIsContractOwner();

        ICreatorTokenTransferValidator validator = getTransferValidator();
        if (address(validator) == address(0)) {
            revert CreatorTokenBase__SetTransferValidatorFirst();
        }

        validator.setTransferSecurityLevelOfCollection(address(this), level);
        validator.setOperatorWhitelistOfCollection(address(this), operatorWhitelistId);
        validator.setPermittedContractReceiverAllowlistOfCollection(address(this), permittedContractReceiversAllowlistId);
    }

    /**
     * @notice Sets the transfer validator for the token contract.
     *
     * @dev    Throws when provided validator contract is not the zero address and doesn't support 
     *         the ICreatorTokenTransferValidator interface. 
     * @dev    Throws when the caller is not the contract owner.
     *
     * @dev    <h4>Postconditions:</h4>
     *         1. The transferValidator address is updated.
     *         2. The `TransferValidatorUpdated` event is emitted.
     *
     * @param transferValidator_ The address of the transfer validator contract.
     */
    function setTransferValidator(address transferValidator_) public {
        _requireCallerIsContractOwner();

        bool isValidTransferValidator = false;

        if(transferValidator_.code.length > 0) {
            try IERC165(transferValidator_).supportsInterface(type(ICreatorTokenTransferValidator).interfaceId) 
                returns (bool supportsInterface) {
                isValidTransferValidator = supportsInterface;
            } catch {}
        }

        if(transferValidator_ != address(0) && !isValidTransferValidator) {
            revert CreatorTokenBase__InvalidTransferValidatorContract();
        }

        emit TransferValidatorUpdated(address(transferValidator), transferValidator_);

        transferValidator = ICreatorTokenTransferValidator(transferValidator_);
    }

    /**
     * @notice Returns the transfer validator contract address for this token contract.
     */
    function getTransferValidator() public view override returns (ICreatorTokenTransferValidator) {
        return transferValidator;
    }

    /**
     * @notice Returns the security policy for this token contract, which includes:
     *         Transfer security level, operator whitelist id, permitted contract receiver allowlist id.
     */
    function getSecurityPolicy() public view override returns (CollectionSecurityPolicy memory) {
        if (address(transferValidator) != address(0)) {
            return transferValidator.getCollectionSecurityPolicy(address(this));
        }

        return CollectionSecurityPolicy({
            transferSecurityLevel: TransferSecurityLevels.Zero,
            operatorWhitelistId: 0,
            permittedContractReceiversId: 0
        });
    }

    /**
     * @notice Returns the list of all whitelisted operators for this token contract.
     * @dev    This can be an expensive call and should only be used in view-only functions.
     */
    function getWhitelistedOperators() public view override returns (address[] memory) {
        if (address(transferValidator) != address(0)) {
            return transferValidator.getWhitelistedOperators(
                transferValidator.getCollectionSecurityPolicy(address(this)).operatorWhitelistId);
        }

        return new address[](0);
    }

    /**
     * @notice Returns the list of permitted contract receivers for this token contract.
     * @dev    This can be an expensive call and should only be used in view-only functions.
     */
    function getPermittedContractReceivers() public view override returns (address[] memory) {
        if (address(transferValidator) != address(0)) {
            return transferValidator.getPermittedContractReceivers(
                transferValidator.getCollectionSecurityPolicy(address(this)).permittedContractReceiversId);
        }

        return new address[](0);
    }

    /**
     * @notice Checks if an operator is whitelisted for this token contract.
     * @param operator The address of the operator to check.
     */
    function isOperatorWhitelisted(address operator) public view override returns (bool) {
        if (address(transferValidator) != address(0)) {
            return transferValidator.isOperatorWhitelisted(
                transferValidator.getCollectionSecurityPolicy(address(this)).operatorWhitelistId, operator);
        }

        return false;
    }

    /**
     * @notice Checks if a contract receiver is permitted for this token contract.
     * @param receiver The address of the receiver to check.
     */
    function isContractReceiverPermitted(address receiver) public view override returns (bool) {
        if (address(transferValidator) != address(0)) {
            return transferValidator.isContractReceiverPermitted(
                transferValidator.getCollectionSecurityPolicy(address(this)).permittedContractReceiversId, receiver);
        }

        return false;
    }

    /**
     * @notice Determines if a transfer is allowed based on the token contract's security policy.  Use this function
     *         to simulate whether or not a transfer made by the specified `caller` from the `from` address to the `to`
     *         address would be allowed by this token's security policy.
     *
     * @notice This function only checks the security policy restrictions and does not check whether token ownership
     *         or approvals are in place. 
     *
     * @param caller The address of the simulated caller.
     * @param from   The address of the sender.
     * @param to     The address of the receiver.
     * @return       True if the transfer is allowed, false otherwise.
     */
    function isTransferAllowed(address caller, address from, address to) public view override returns (bool) {
        if (address(transferValidator) != address(0)) {
            try transferValidator.applyCollectionTransferPolicy(caller, from, to) {
                return true;
            } catch {
                return false;
            }
        }
        return true;
    }

    /**
     * @dev Pre-validates a token transfer, reverting if the transfer is not allowed by this token's security policy.
     *      Inheriting contracts are responsible for overriding the _beforeTokenTransfer function, or its equivalent
     *      and calling _validateBeforeTransfer so that checks can be properly applied during token transfers.
     *
     * @dev Throws when the transfer doesn't comply with the collection's transfer policy, if the transferValidator is
     *      set to a non-zero address.
     *
     * @param caller  The address of the caller.
     * @param from    The address of the sender.
     * @param to      The address of the receiver.
     */
    function _preValidateTransfer(
        address caller, 
        address from, 
        address to, 
        uint256 /*tokenId*/, 
        uint256 /*value*/) internal virtual override {
        if (address(transferValidator) != address(0)) {
            transferValidator.applyCollectionTransferPolicy(caller, from, to);
        }
    }
}

// File: @openzeppelin/contracts/interfaces/IERC2981.sol


// OpenZeppelin Contracts (last updated v5.0.0) (interfaces/IERC2981.sol)

pragma solidity ^0.8.20;


/**
 * @dev Interface for the NFT Royalty Standard.
 *
 * A standardized way to retrieve royalty payment information for non-fungible tokens (NFTs) to enable universal
 * support for royalty payments across all NFT marketplaces and ecosystem participants.
 */
interface IERC2981 is IERC165 {
    /**
     * @dev Returns how much royalty is owed and to whom, based on a sale price that may be denominated in any unit of
     * exchange. The royalty amount is denominated and should be paid in that same unit of exchange.
     */
    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) external view returns (address receiver, uint256 royaltyAmount);
}

// File: @openzeppelin/contracts/utils/introspection/ERC165.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;


/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 */
abstract contract ERC165 is IERC165 {
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}

// File: @openzeppelin/contracts/token/common/ERC2981.sol


// OpenZeppelin Contracts (last updated v5.0.0) (token/common/ERC2981.sol)

pragma solidity ^0.8.20;



/**
 * @dev Implementation of the NFT Royalty Standard, a standardized way to retrieve royalty payment information.
 *
 * Royalty information can be specified globally for all token ids via {_setDefaultRoyalty}, and/or individually for
 * specific token ids via {_setTokenRoyalty}. The latter takes precedence over the first.
 *
 * Royalty is specified as a fraction of sale price. {_feeDenominator} is overridable but defaults to 10000, meaning the
 * fee is specified in basis points by default.
 *
 * IMPORTANT: ERC-2981 only specifies a way to signal royalty information and does not enforce its payment. See
 * https://eips.ethereum.org/EIPS/eip-2981#optional-royalty-payments[Rationale] in the EIP. Marketplaces are expected to
 * voluntarily pay royalties together with sales, but note that this standard is not yet widely supported.
 */
abstract contract ERC2981 is IERC2981, ERC165 {
    struct RoyaltyInfo {
        address receiver;
        uint96 royaltyFraction;
    }

    RoyaltyInfo private _defaultRoyaltyInfo;
    mapping(uint256 tokenId => RoyaltyInfo) private _tokenRoyaltyInfo;

    /**
     * @dev The default royalty set is invalid (eg. (numerator / denominator) >= 1).
     */
    error ERC2981InvalidDefaultRoyalty(uint256 numerator, uint256 denominator);

    /**
     * @dev The default royalty receiver is invalid.
     */
    error ERC2981InvalidDefaultRoyaltyReceiver(address receiver);

    /**
     * @dev The royalty set for an specific `tokenId` is invalid (eg. (numerator / denominator) >= 1).
     */
    error ERC2981InvalidTokenRoyalty(uint256 tokenId, uint256 numerator, uint256 denominator);

    /**
     * @dev The royalty receiver for `tokenId` is invalid.
     */
    error ERC2981InvalidTokenRoyaltyReceiver(uint256 tokenId, address receiver);

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override(IERC165, ERC165) returns (bool) {
        return interfaceId == type(IERC2981).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @inheritdoc IERC2981
     */
    function royaltyInfo(uint256 tokenId, uint256 salePrice) public view virtual returns (address, uint256) {
        RoyaltyInfo memory royalty = _tokenRoyaltyInfo[tokenId];

        if (royalty.receiver == address(0)) {
            royalty = _defaultRoyaltyInfo;
        }

        uint256 royaltyAmount = (salePrice * royalty.royaltyFraction) / _feeDenominator();

        return (royalty.receiver, royaltyAmount);
    }

    /**
     * @dev The denominator with which to interpret the fee set in {_setTokenRoyalty} and {_setDefaultRoyalty} as a
     * fraction of the sale price. Defaults to 10000 so fees are expressed in basis points, but may be customized by an
     * override.
     */
    function _feeDenominator() internal pure virtual returns (uint96) {
        return 10000;
    }

    /**
     * @dev Sets the royalty information that all ids in this contract will default to.
     *
     * Requirements:
     *
     * - `receiver` cannot be the zero address.
     * - `feeNumerator` cannot be greater than the fee denominator.
     */
    function _setDefaultRoyalty(address receiver, uint96 feeNumerator) internal virtual {
        uint256 denominator = _feeDenominator();
        if (feeNumerator > denominator) {
            // Royalty fee will exceed the sale price
            revert ERC2981InvalidDefaultRoyalty(feeNumerator, denominator);
        }
        if (receiver == address(0)) {
            revert ERC2981InvalidDefaultRoyaltyReceiver(address(0));
        }

        _defaultRoyaltyInfo = RoyaltyInfo(receiver, feeNumerator);
    }

    /**
     * @dev Removes default royalty information.
     */
    function _deleteDefaultRoyalty() internal virtual {
        delete _defaultRoyaltyInfo;
    }

    /**
     * @dev Sets the royalty information for a specific token id, overriding the global default.
     *
     * Requirements:
     *
     * - `receiver` cannot be the zero address.
     * - `feeNumerator` cannot be greater than the fee denominator.
     */
    function _setTokenRoyalty(uint256 tokenId, address receiver, uint96 feeNumerator) internal virtual {
        uint256 denominator = _feeDenominator();
        if (feeNumerator > denominator) {
            // Royalty fee will exceed the sale price
            revert ERC2981InvalidTokenRoyalty(tokenId, feeNumerator, denominator);
        }
        if (receiver == address(0)) {
            revert ERC2981InvalidTokenRoyaltyReceiver(tokenId, address(0));
        }

        _tokenRoyaltyInfo[tokenId] = RoyaltyInfo(receiver, feeNumerator);
    }

    /**
     * @dev Resets royalty information for the token id back to the global default.
     */
    function _resetTokenRoyalty(uint256 tokenId) internal virtual {
        delete _tokenRoyaltyInfo[tokenId];
    }
}

// File: https://github.com/limitbreakinc/creator-token-contracts/contracts/programmable-royalties/BasicRoyalties.sol


pragma solidity ^0.8.4;


/**
 * @title BasicRoyaltiesBase
 * @author Limit Break, Inc.
 * @dev Base functionality of an NFT mix-in contract implementing the most basic form of programmable royalties.
 */
abstract contract BasicRoyaltiesBase is ERC2981 {

    event DefaultRoyaltySet(address indexed receiver, uint96 feeNumerator);
    event TokenRoyaltySet(uint256 indexed tokenId, address indexed receiver, uint96 feeNumerator);

    function _setDefaultRoyalty(address receiver, uint96 feeNumerator) internal virtual override {
        super._setDefaultRoyalty(receiver, feeNumerator);
        emit DefaultRoyaltySet(receiver, feeNumerator);
    }

    function _setTokenRoyalty(uint256 tokenId, address receiver, uint96 feeNumerator) internal virtual override {
        super._setTokenRoyalty(tokenId, receiver, feeNumerator);
        emit TokenRoyaltySet(tokenId, receiver, feeNumerator);
    }
}

/**
 * @title BasicRoyalties
 * @author Limit Break, Inc.
 * @notice Constructable BasicRoyalties Contract implementation.
 */
abstract contract BasicRoyalties is BasicRoyaltiesBase {
    constructor(address receiver, uint96 feeNumerator) {
        _setDefaultRoyalty(receiver, feeNumerator);
    }
}

/**
 * @title BasicRoyaltiesInitializable
 * @author Limit Break, Inc.
 * @notice Initializable BasicRoyalties Contract implementation to allow for EIP-1167 clones. 
 */
abstract contract BasicRoyaltiesInitializable is BasicRoyaltiesBase {}
// File: DN404SMirror.sol


pragma solidity ^0.8.25;




contract DN404SMirror is DN404Mirror, CreatorTokenBase, BasicRoyalties {
    error UnauthorizedAccount(address account);

    constructor(
        address _royaltiesRecipientAddress,
        uint96 _royaltyPercentageBPS
    )
        DN404Mirror(tx.origin)
        BasicRoyalties(_royaltiesRecipientAddress, _royaltyPercentageBPS)
    {}

    /// override required by solidity
    function _requireCallerIsContractOwner() internal view override {
        if (_msgSender() != owner()) {
            revert UnauthorizedAccount(_msgSender());
        }
    }

    function transferFrom(
        address from,
        address to,
        uint256 id
    ) public payable override(DN404Mirror) {
        _validateBeforeTransfer(from, to, id);
        super.transferFrom(from, to, id);
        _validateAfterTransfer(from, to, id);
    }

    /// function to check if the contract supports an interface
    /// override required by solidity
    function supportsInterface(bytes4 interfaceId)
        public
        view
        virtual
        override(DN404Mirror, ERC2981)
        returns (bool)
    {
        return
            interfaceId == type(ICreatorToken).interfaceId ||
            super.supportsInterface(interfaceId);
    }
}