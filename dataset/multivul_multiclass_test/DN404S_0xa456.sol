// SPDX-License-Identifier: UNLICENSED
// File: dn404/src/DN404.sol


pragma solidity ^0.8.4;

/// @title DN404S
/// @notice DN404 is a hybrid ERC20 and ERC721 implementation that mints
/// and burns NFTs based on an account's ERC20 token balance.
///
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
/// @dev Note:
/// - The ERC721 data is stored in this base DN404 contract, however a
///   DN404Mirror contract ***MUST*** be deployed and linked during
///   initialization.
/// - For ERC20 transfers, the most recently acquired NFT will be burned / transferred out first.
/// - A unit worth of ERC20 tokens equates to a deed to one NFT token.
///   The skip NFT status determines if this deed is automatically exercised.
///   An account can configure their skip NFT status.
///     * If `getSkipNFT(owner) == true`, ERC20 mints / transfers to `owner`
///       will NOT trigger NFT mints / transfers to `owner` (i.e. deeds are left unexercised).
///     * If `getSkipNFT(owner) == false`, ERC20 mints / transfers to `owner`
///       will trigger NFT mints / transfers to `owner`, until the NFT balance of `owner`
///       is equal to its ERC20 balance divided by the unit (rounded down).
/// - Invariant: `mirror.balanceOf(owner) <= base.balanceOf(owner) / _unit()`.
/// - The gas costs for automatic minting / transferring / burning of NFTs is O(n).
///   This can exceed the block gas limit.
///   Applications and users may need to break up large transfers into a few transactions.
/// - This implementation does not support "safe" transfers for automatic NFT transfers.
/// - The ERC20 token allowances and ERC721 token / operator approvals are separate.
/// - For MEV safety, users should NOT have concurrently open orders for the ERC20 and ERC721.
abstract contract DN404 {
    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                           EVENTS                           */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Emitted when `amount` tokens is transferred from `from` to `to`.
    event Transfer(address indexed from, address indexed to, uint256 amount);

    /// @dev Emitted when `amount` tokens is approved by `owner` to be used by `spender`.
    event Approval(address indexed owner, address indexed spender, uint256 amount);

    /// @dev Emitted when `owner` sets their skipNFT flag to `status`.
    event SkipNFTSet(address indexed owner, bool status);

    /// @dev `keccak256(bytes("Transfer(address,address,uint256)"))`.
    uint256 private constant _TRANSFER_EVENT_SIGNATURE =
        0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef;

    /// @dev `keccak256(bytes("Approval(address,address,uint256)"))`.
    uint256 private constant _APPROVAL_EVENT_SIGNATURE =
        0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925;

    /// @dev `keccak256(bytes("SkipNFTSet(address,bool)"))`.
    uint256 private constant _SKIP_NFT_SET_EVENT_SIGNATURE =
        0xb5a1de456fff688115a4f75380060c23c8532d14ff85f687cc871456d6420393;

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                        CUSTOM ERRORS                       */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Thrown when attempting to double-initialize the contract.
    error DNAlreadyInitialized();

    /// @dev The function can only be called after the contract has been initialized.
    error DNNotInitialized();

    /// @dev Thrown when attempting to transfer or burn more tokens than sender's balance.
    error InsufficientBalance();

    /// @dev Thrown when a spender attempts to transfer tokens with an insufficient allowance.
    error InsufficientAllowance();

    /// @dev Thrown when minting an amount of tokens that would overflow the max tokens.
    error TotalSupplyOverflow();

    /// @dev The unit must be greater than zero and less than `2**96`.
    error InvalidUnit();

    /// @dev Thrown when the caller for a fallback NFT function is not the mirror contract.
    error SenderNotMirror();

    /// @dev Thrown when attempting to transfer tokens to the zero address.
    error TransferToZeroAddress();

    /// @dev Thrown when the mirror address provided for initialization is the zero address.
    error MirrorAddressIsZero();

    /// @dev Thrown when the link call to the mirror contract reverts.
    error LinkMirrorContractFailed();

    /// @dev Thrown when setting an NFT token approval
    /// and the caller is not the owner or an approved operator.
    error ApprovalCallerNotOwnerNorApproved();

    /// @dev Thrown when transferring an NFT
    /// and the caller is not the owner or an approved operator.
    error TransferCallerNotOwnerNorApproved();

    /// @dev Thrown when transferring an NFT and the from address is not the current owner.
    error TransferFromIncorrectOwner();

    /// @dev Thrown when checking the owner or approved address for a non-existent NFT.
    error TokenDoesNotExist();

    /// @dev The function selector is not recognized.
    error FnSelectorNotRecognized();

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                         CONSTANTS                          */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev The flag to denote that the skip NFT flag is initialized.
    uint8 internal constant _ADDRESS_DATA_SKIP_NFT_INITIALIZED_FLAG = 1 << 0;

    /// @dev The flag to denote that the address should skip NFTs.
    uint8 internal constant _ADDRESS_DATA_SKIP_NFT_FLAG = 1 << 1;

    /// @dev The flag to denote that the address has overridden the default Permit2 allowance.
    uint8 internal constant _ADDRESS_DATA_OVERRIDE_PERMIT2_FLAG = 1 << 2;

    /// @dev The canonical Permit2 address.
    /// For signature-based allowance granting for single transaction ERC20 `transferFrom`.
    /// To enable, override `_givePermit2DefaultInfiniteAllowance()`.
    /// [Github](https://github.com/Uniswap/permit2)
    /// [Etherscan](https://etherscan.io/address/0x000000000022D473030F116dDEE9F6B43aC78BA3)
    address internal constant _PERMIT2 = 0x000000000022D473030F116dDEE9F6B43aC78BA3;

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                          STORAGE                           */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Struct containing an address's token data and settings.
    struct AddressData {
        // Auxiliary data.
        uint88 aux;
        // Flags for `initialized` and `skipNFT`.
        uint8 flags;
        // The alias for the address. Zero means absence of an alias.
        uint32 addressAlias;
        // The number of NFT tokens.
        uint32 ownedLength;
        // The token balance in wei.
        uint96 balance;
    }

    /// @dev A uint32 map in storage.
    struct Uint32Map {
        uint256 spacer;
    }

    /// @dev A bitmap in storage.
    struct Bitmap {
        uint256 spacer;
    }

    /// @dev A struct to wrap a uint256 in storage.
    struct Uint256Ref {
        uint256 value;
    }

    /// @dev A mapping of an address pair to a Uint256Ref.
    struct AddressPairToUint256RefMap {
        uint256 spacer;
    }

    /// @dev Struct containing the base token contract storage.
    struct DN404Storage {
        // Current number of address aliases assigned.
        uint32 numAliases;
        // Next NFT ID to assign for a mint.
        uint32 nextTokenId;
        // The head of the burned pool.
        uint32 burnedPoolHead;
        // The tail of the burned pool.
        uint32 burnedPoolTail;
        // Total number of NFTs in existence.
        uint32 totalNFTSupply;
        // Total supply of tokens.
        uint96 totalSupply;
        // Address of the NFT mirror contract.
        address mirrorERC721;
        // Mapping of a user alias number to their address.
        mapping(uint32 => address) aliasToAddress;
        // Mapping of user operator approvals for NFTs.
        AddressPairToUint256RefMap operatorApprovals;
        // Mapping of NFT approvals to approved operators.
        mapping(uint256 => address) nftApprovals;
        // Bitmap of whether an non-zero NFT approval may exist.
        Bitmap mayHaveNFTApproval;
        // Bitmap of whether a NFT ID exists. Ignored if `_useExistsLookup()` returns false.
        Bitmap exists;
        // Mapping of user allowances for ERC20 spenders.
        AddressPairToUint256RefMap allowance;
        // Mapping of NFT IDs owned by an address.
        mapping(address => Uint32Map) owned;
        // The pool of burned NFT IDs.
        Uint32Map burnedPool;
        // Even indices: owner aliases. Odd indices: owned indices.
        Uint32Map oo;
        // Mapping of user account AddressData.
        mapping(address => AddressData) addressData;
    }

    /// @dev Returns a storage pointer for DN404Storage.
    function _getDN404Storage() internal pure virtual returns (DN404Storage storage $) {
        /// @solidity memory-safe-assembly
        assembly {
            // `uint72(bytes9(keccak256("DN404_STORAGE")))`.
            $.slot := 0xa20d6e21d0e5255308 // Truncate to 9 bytes to reduce bytecode size.
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                         INITIALIZER                        */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Initializes the DN404 contract with an
    /// `initialTokenSupply`, `initialTokenOwner` and `mirror` NFT contract address.
    ///
    /// Note: The `initialSupplyOwner` will have their skip NFT status set to true.
    function _initializeDN404(
        uint256 initialTokenSupply,
        address initialSupplyOwner,
        address mirror
    ) internal virtual {
        DN404Storage storage $ = _getDN404Storage();

        unchecked {
            if (_unit() - 1 >= 2 ** 96 - 1) revert InvalidUnit();
        }
        if ($.mirrorERC721 != address(0)) revert DNAlreadyInitialized();
        if (mirror == address(0)) revert MirrorAddressIsZero();

        /// @solidity memory-safe-assembly
        assembly {
            // Make the call to link the mirror contract.
            mstore(0x00, 0x0f4599e5) // `linkMirrorContract(address)`.
            mstore(0x20, caller())
            if iszero(and(eq(mload(0x00), 1), call(gas(), mirror, 0, 0x1c, 0x24, 0x00, 0x20))) {
                mstore(0x00, 0xd125259c) // `LinkMirrorContractFailed()`.
                revert(0x1c, 0x04)
            }
        }

        $.nextTokenId = uint32(_toUint(_useOneIndexed()));
        $.mirrorERC721 = mirror;

        if (initialTokenSupply != 0) {
            if (initialSupplyOwner == address(0)) revert TransferToZeroAddress();
            if (_totalSupplyOverflows(initialTokenSupply)) revert TotalSupplyOverflow();

            $.totalSupply = uint96(initialTokenSupply);
            AddressData storage initialOwnerAddressData = $.addressData[initialSupplyOwner];
            initialOwnerAddressData.balance = uint96(initialTokenSupply);

            /// @solidity memory-safe-assembly
            assembly {
                // Emit the {Transfer} event.
                mstore(0x00, initialTokenSupply)
                log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, 0, shr(96, shl(96, initialSupplyOwner)))
            }

            _setSkipNFT(initialSupplyOwner, true);
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*               BASE UNIT FUNCTION TO OVERRIDE               */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Amount of token balance that is equal to one NFT.
    ///
    /// Note: The return value MUST be kept constant after `_initializeDN404` is called.
    function _unit() internal view virtual returns (uint256) {
        return 10 ** 18;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*               METADATA FUNCTIONS TO OVERRIDE               */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the name of the token.
    function name() public view virtual returns (string memory);

    /// @dev Returns the symbol of the token.
    function symbol() public view virtual returns (string memory);

    /// @dev Returns the Uniform Resource Identifier (URI) for token `id`.
    function _tokenURI(uint256 id) internal view virtual returns (string memory);

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                       CONFIGURABLES                        */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns whether the tokens IDs are from `[1..n]` instead of `[0..n-1]`.
    function _useOneIndexed() internal pure virtual returns (bool) {
        return true;
    }

    /// @dev Returns if direct NFT transfers should be used during ERC20 transfers
    /// whenever possible, instead of burning and re-minting.
    function _useDirectTransfersIfPossible() internal view virtual returns (bool) {
        return true;
    }

    /// @dev Returns if burns should be added to the burn pool.
    /// This returns false by default, which means the NFT IDs are re-minted in a cycle.
    function _addToBurnedPool(uint256 totalNFTSupplyAfterBurn, uint256 totalSupplyAfterBurn)
        internal
        view
        virtual
        returns (bool)
    {
        // Silence unused variable compiler warning.
        totalSupplyAfterBurn = totalNFTSupplyAfterBurn;
        return false;
    }

    /// @dev Returns whether to use the exists bitmap for more efficient
    /// scanning of an empty token ID slot.
    /// Recommended for collections that do not use the burn pool,
    /// and are expected to have nearly all possible NFTs materialized.
    ///
    /// Note: The returned value must be constant after initialization.
    function _useExistsLookup() internal view virtual returns (bool) {
        return true;
    }

    /// @dev Hook that is called after a batch of NFT transfers.
    /// The lengths of `from`, `to`, and `ids` are guaranteed to be the same.
    function _afterNFTTransfers(address[] memory from, address[] memory to, uint256[] memory ids)
        internal
        virtual
    {}

    /// @dev Override this function to return true if `_afterNFTTransfers` is used.
    /// This is to help the compiler avoid producing dead bytecode.
    function _useAfterNFTTransfers() internal virtual returns (bool) {}

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                      ERC20 OPERATIONS                      */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the decimals places of the token. Defaults to 18.
    /// Does not affect DN404's internal calculations.
    /// Will only affect the frontend UI on most protocols.
    function decimals() public view virtual returns (uint8) {
        return 18;
    }

    /// @dev Returns the amount of tokens in existence.
    function totalSupply() public view virtual returns (uint256) {
        return uint256(_getDN404Storage().totalSupply);
    }

    /// @dev Returns the amount of tokens owned by `owner`.
    function balanceOf(address owner) public view virtual returns (uint256) {
        return _getDN404Storage().addressData[owner].balance;
    }

    /// @dev Returns the amount of tokens that `spender` can spend on behalf of `owner`.
    function allowance(address owner, address spender) public view returns (uint256) {
        if (_givePermit2DefaultInfiniteAllowance() && spender == _PERMIT2) {
            uint8 flags = _getDN404Storage().addressData[owner].flags;
            if ((flags & _ADDRESS_DATA_OVERRIDE_PERMIT2_FLAG) == uint256(0)) {
                return type(uint256).max;
            }
        }
        return _ref(_getDN404Storage().allowance, owner, spender).value;
    }

    /// @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
    ///
    /// Emits a {Approval} event.
    function approve(address spender, uint256 amount) public virtual returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }

    /// @dev Transfer `amount` tokens from the caller to `to`.
    ///
    /// Will burn sender NFTs if balance after transfer is less than
    /// the amount required to support the current NFT balance.
    ///
    /// Will mint NFTs to `to` if the recipient's new balance supports
    /// additional NFTs ***AND*** the `to` address's skipNFT flag is
    /// set to false.
    ///
    /// Requirements:
    /// - `from` must at least have `amount`.
    ///
    /// Emits a {Transfer} event.
    function transfer(address to, uint256 amount) public virtual returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    /// @dev Transfers `amount` tokens from `from` to `to`.
    ///
    /// Note: Does not update the allowance if it is the maximum uint256 value.
    ///
    /// Will burn sender NFTs if balance after transfer is less than
    /// the amount required to support the current NFT balance.
    ///
    /// Will mint NFTs to `to` if the recipient's new balance supports
    /// additional NFTs ***AND*** the `to` address's skipNFT flag is
    /// set to false.
    ///
    /// Requirements:
    /// - `from` must at least have `amount`.
    /// - The caller must have at least `amount` of allowance to transfer the tokens of `from`.
    ///
    /// Emits a {Transfer} event.
    function transferFrom(address from, address to, uint256 amount) public virtual returns (bool) {
        Uint256Ref storage a = _ref(_getDN404Storage().allowance, from, msg.sender);

        uint256 allowed = _givePermit2DefaultInfiniteAllowance() && msg.sender == _PERMIT2
            && (_getDN404Storage().addressData[from].flags & _ADDRESS_DATA_OVERRIDE_PERMIT2_FLAG)
                == uint256(0) ? type(uint256).max : a.value;

        if (allowed != type(uint256).max) {
            if (amount > allowed) revert InsufficientAllowance();
            unchecked {
                a.value = allowed - amount;
            }
        }
        _transfer(from, to, amount);
        return true;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                          PERMIT2                           */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Whether Permit2 has infinite allowances by default for all owners.
    /// For signature-based allowance granting for single transaction ERC20 `transferFrom`.
    /// To enable, override this function to return true.
    ///
    /// Note: The returned value SHOULD be kept constant.
    /// If the returned value changes from false to true,
    /// it can override the user customized allowances for Permit2 to infinity.
    function _givePermit2DefaultInfiniteAllowance() internal view virtual returns (bool) {
        return false;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                  INTERNAL MINT FUNCTIONS                   */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Mints `amount` tokens to `to`, increasing the total supply.
    ///
    /// Will mint NFTs to `to` if the recipient's new balance supports
    /// additional NFTs ***AND*** the `to` address's skipNFT flag is set to false.
    ///
    /// Note:
    /// - May mint more NFTs than `amount / _unit()`.
    ///   The number of NFTs minted is what is needed to make `to`'s NFT balance whole.
    /// - Token IDs wraps back to `_toUint(_useOneIndexed())` upon exceeding the upper limit.
    ///
    /// Emits a {Transfer} event.
    function _mint(address to, uint256 amount) internal virtual {
        if (to == address(0)) revert TransferToZeroAddress();

        DN404Storage storage $ = _getDN404Storage();
        if ($.mirrorERC721 == address(0)) revert DNNotInitialized();
        AddressData storage toAddressData = $.addressData[to];

        _DNMintTemps memory t;
        unchecked {
            {
                uint256 toBalance = uint256(toAddressData.balance) + amount;
                toAddressData.balance = uint96(toBalance);
                t.toEnd = toBalance / _unit();
            }
            uint256 idLimit;
            {
                uint256 newTotalSupply = uint256($.totalSupply) + amount;
                $.totalSupply = uint96(newTotalSupply);
                uint256 overflows = _toUint(_totalSupplyOverflows(newTotalSupply));
                if (overflows | _toUint(newTotalSupply < amount) != 0) revert TotalSupplyOverflow();
                idLimit = newTotalSupply / _unit();
            }
            while (!getSkipNFT(to)) {
                Uint32Map storage toOwned = $.owned[to];
                Uint32Map storage oo = $.oo;
                uint256 toIndex = toAddressData.ownedLength;
                if ((t.numNFTMints = _zeroFloorSub(t.toEnd, toIndex)) == uint256(0)) break;

                t.packedLogs = _packedLogsMalloc(t.numNFTMints);
                _packedLogsSet(t.packedLogs, to, 0);
                $.totalNFTSupply += uint32(t.numNFTMints);
                toAddressData.ownedLength = uint32(t.toEnd);
                t.toAlias = _registerAndResolveAlias(toAddressData, to);
                uint32 burnedPoolHead = $.burnedPoolHead;
                t.burnedPoolTail = $.burnedPoolTail;
                t.nextTokenId = _wrapNFTId($.nextTokenId, idLimit);
                // Mint loop.
                do {
                    uint256 id;
                    if (burnedPoolHead != t.burnedPoolTail) {
                        id = _get($.burnedPool, burnedPoolHead++);
                    } else {
                        id = t.nextTokenId;
                        while (_get(oo, _ownershipIndex(id)) != 0) {
                            id = _useExistsLookup()
                                ? _wrapNFTId(_findFirstUnset($.exists, id + 1, idLimit), idLimit)
                                : _wrapNFTId(id + 1, idLimit);
                        }
                        t.nextTokenId = _wrapNFTId(id + 1, idLimit);
                    }
                    if (_useExistsLookup()) _set($.exists, id, true);
                    _set(toOwned, toIndex, uint32(id));
                    _setOwnerAliasAndOwnedIndex(oo, id, t.toAlias, uint32(toIndex++));
                    _packedLogsAppend(t.packedLogs, id);
                } while (toIndex != t.toEnd);

                $.nextTokenId = uint32(t.nextTokenId);
                $.burnedPoolHead = burnedPoolHead;
                _packedLogsSend(t.packedLogs, $);
                break;
            }
        }
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Transfer} event.
            mstore(0x00, amount)
            log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, 0, shr(96, shl(96, to)))
        }
        if (_useAfterNFTTransfers()) {
            _afterNFTTransfers(
                _zeroAddresses(t.numNFTMints),
                _filled(t.numNFTMints, to),
                _packedLogsIds(t.packedLogs)
            );
        }
    }

    /// @dev Mints `amount` tokens to `to`, increasing the total supply.
    /// This variant mints NFT tokens starting from ID
    /// `preTotalSupply / _unit() + _toUint(_useOneIndexed())`.
    /// The `nextTokenId` will not be changed.
    /// If any NFTs are minted, the burned pool will be invalidated (emptied).
    ///
    /// Will mint NFTs to `to` if the recipient's new balance supports
    /// additional NFTs ***AND*** the `to` address's skipNFT flag is set to false.
    ///
    /// Note:
    /// - May mint more NFTs than `amount / _unit()`.
    ///   The number of NFTs minted is what is needed to make `to`'s NFT balance whole.
    /// - Token IDs wraps back to `_toUint(_useOneIndexed())` upon exceeding the upper limit.
    ///
    /// Emits a {Transfer} event.
    function _mintNext(address to, uint256 amount) internal virtual {
        if (to == address(0)) revert TransferToZeroAddress();

        DN404Storage storage $ = _getDN404Storage();
        if ($.mirrorERC721 == address(0)) revert DNNotInitialized();
        AddressData storage toAddressData = $.addressData[to];

        _DNMintTemps memory t;
        unchecked {
            {
                uint256 toBalance = uint256(toAddressData.balance) + amount;
                toAddressData.balance = uint96(toBalance);
                t.toEnd = toBalance / _unit();
            }
            uint256 id;
            uint256 idLimit;
            {
                uint256 preTotalSupply = uint256($.totalSupply);
                uint256 newTotalSupply = uint256(preTotalSupply) + amount;
                $.totalSupply = uint96(newTotalSupply);
                uint256 overflows = _toUint(_totalSupplyOverflows(newTotalSupply));
                if (overflows | _toUint(newTotalSupply < amount) != 0) revert TotalSupplyOverflow();
                idLimit = newTotalSupply / _unit();
                id = _wrapNFTId(preTotalSupply / _unit() + _toUint(_useOneIndexed()), idLimit);
            }
            while (!getSkipNFT(to)) {
                Uint32Map storage toOwned = $.owned[to];
                Uint32Map storage oo = $.oo;
                uint256 toIndex = toAddressData.ownedLength;
                if ((t.numNFTMints = _zeroFloorSub(t.toEnd, toIndex)) == uint256(0)) break;

                t.packedLogs = _packedLogsMalloc(t.numNFTMints);
                // Invalidate (empty) the burned pool.
                $.burnedPoolHead = 0;
                $.burnedPoolTail = 0;
                _packedLogsSet(t.packedLogs, to, 0);
                $.totalNFTSupply += uint32(t.numNFTMints);
                toAddressData.ownedLength = uint32(t.toEnd);
                t.toAlias = _registerAndResolveAlias(toAddressData, to);
                // Mint loop.
                do {
                    while (_get(oo, _ownershipIndex(id)) != 0) {
                        id = _useExistsLookup()
                            ? _wrapNFTId(_findFirstUnset($.exists, id + 1, idLimit), idLimit)
                            : _wrapNFTId(id + 1, idLimit);
                    }
                    if (_useExistsLookup()) _set($.exists, id, true);
                    _set(toOwned, toIndex, uint32(id));
                    _setOwnerAliasAndOwnedIndex(oo, id, t.toAlias, uint32(toIndex++));
                    _packedLogsAppend(t.packedLogs, id);
                    id = _wrapNFTId(id + 1, idLimit);
                } while (toIndex != t.toEnd);

                _packedLogsSend(t.packedLogs, $);
                break;
            }
        }
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Transfer} event.
            mstore(0x00, amount)
            log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, 0, shr(96, shl(96, to)))
        }
        if (_useAfterNFTTransfers()) {
            _afterNFTTransfers(
                _zeroAddresses(t.numNFTMints),
                _filled(t.numNFTMints, to),
                _packedLogsIds(t.packedLogs)
            );
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                  INTERNAL BURN FUNCTIONS                   */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Burns `amount` tokens from `from`, reducing the total supply.
    ///
    /// Will burn sender NFTs if balance after transfer is less than
    /// the amount required to support the current NFT balance.
    ///
    /// Emits a {Transfer} event.
    function _burn(address from, uint256 amount) internal virtual {
        DN404Storage storage $ = _getDN404Storage();
        if ($.mirrorERC721 == address(0)) revert DNNotInitialized();

        AddressData storage fromAddressData = $.addressData[from];
        _DNBurnTemps memory t;

        unchecked {
            t.fromBalance = fromAddressData.balance;
            if (amount > t.fromBalance) revert InsufficientBalance();

            fromAddressData.balance = uint96(t.fromBalance -= amount);
            t.totalSupply = uint256($.totalSupply) - amount;
            $.totalSupply = uint96(t.totalSupply);

            Uint32Map storage fromOwned = $.owned[from];
            uint256 fromIndex = fromAddressData.ownedLength;
            t.numNFTBurns = _zeroFloorSub(fromIndex, t.fromBalance / _unit());

            if (t.numNFTBurns != 0) {
                t.packedLogs = _packedLogsMalloc(t.numNFTBurns);
                _packedLogsSet(t.packedLogs, from, 1);
                bool addToBurnedPool;
                {
                    uint256 totalNFTSupply = uint256($.totalNFTSupply) - t.numNFTBurns;
                    $.totalNFTSupply = uint32(totalNFTSupply);
                    addToBurnedPool = _addToBurnedPool(totalNFTSupply, t.totalSupply);
                }

                Uint32Map storage oo = $.oo;
                uint256 fromEnd = fromIndex - t.numNFTBurns;
                fromAddressData.ownedLength = uint32(fromEnd);
                uint32 burnedPoolTail = $.burnedPoolTail;
                // Burn loop.
                do {
                    uint256 id = _get(fromOwned, --fromIndex);
                    _setOwnerAliasAndOwnedIndex(oo, id, 0, 0);
                    _packedLogsAppend(t.packedLogs, id);
                    if (_useExistsLookup()) _set($.exists, id, false);
                    if (addToBurnedPool) _set($.burnedPool, burnedPoolTail++, uint32(id));
                    if (_get($.mayHaveNFTApproval, id)) {
                        _set($.mayHaveNFTApproval, id, false);
                        delete $.nftApprovals[id];
                    }
                } while (fromIndex != fromEnd);

                if (addToBurnedPool) $.burnedPoolTail = burnedPoolTail;
                _packedLogsSend(t.packedLogs, $);
            }
        }
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Transfer} event.
            mstore(0x00, amount)
            log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, shr(96, shl(96, from)), 0)
        }
        if (_useAfterNFTTransfers()) {
            _afterNFTTransfers(
                _filled(t.numNFTBurns, from),
                _zeroAddresses(t.numNFTBurns),
                _packedLogsIds(t.packedLogs)
            );
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                INTERNAL TRANSFER FUNCTIONS                 */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Moves `amount` of tokens from `from` to `to`.
    ///
    /// Will burn sender NFTs if balance after transfer is less than
    /// the amount required to support the current NFT balance.
    ///
    /// Will mint NFTs to `to` if the recipient's new balance supports
    /// additional NFTs ***AND*** the `to` address's skipNFT flag is
    /// set to false.
    ///
    /// Emits a {Transfer} event.
    function _transfer(address from, address to, uint256 amount) internal virtual {
        if (to == address(0)) revert TransferToZeroAddress();

        DN404Storage storage $ = _getDN404Storage();
        AddressData storage fromAddressData = $.addressData[from];
        AddressData storage toAddressData = $.addressData[to];
        if ($.mirrorERC721 == address(0)) revert DNNotInitialized();

        _DNTransferTemps memory t;
        t.fromOwnedLength = fromAddressData.ownedLength;
        t.toOwnedLength = toAddressData.ownedLength;

        unchecked {
            {
                uint256 fromBalance = fromAddressData.balance;
                if (amount > fromBalance) revert InsufficientBalance();
                fromAddressData.balance = uint96(fromBalance -= amount);

                uint256 toBalance = uint256(toAddressData.balance) + amount;
                toAddressData.balance = uint96(toBalance);
                t.numNFTBurns = _zeroFloorSub(t.fromOwnedLength, fromBalance / _unit());

                if (!getSkipNFT(to)) {
                    if (from == to) t.toOwnedLength = t.fromOwnedLength - t.numNFTBurns;
                    t.numNFTMints = _zeroFloorSub(toBalance / _unit(), t.toOwnedLength);
                }
            }

            while (_useDirectTransfersIfPossible()) {
                uint256 n = _min(t.fromOwnedLength, _min(t.numNFTBurns, t.numNFTMints));
                if (n == uint256(0)) break;
                t.numNFTBurns -= n;
                t.numNFTMints -= n;
                if (from == to) {
                    t.toOwnedLength += n;
                    break;
                }
                t.directLogs = _directLogsMalloc(n, from, to);
                Uint32Map storage fromOwned = $.owned[from];
                Uint32Map storage toOwned = $.owned[to];
                t.toAlias = _registerAndResolveAlias(toAddressData, to);
                uint256 toIndex = t.toOwnedLength;
                n = toIndex + n;
                // Direct transfer loop.
                do {
                    uint256 id = _get(fromOwned, --t.fromOwnedLength);
                    _set(toOwned, toIndex, uint32(id));
                    _setOwnerAliasAndOwnedIndex($.oo, id, t.toAlias, uint32(toIndex));
                    _directLogsAppend(t.directLogs, id);
                    if (_get($.mayHaveNFTApproval, id)) {
                        _set($.mayHaveNFTApproval, id, false);
                        delete $.nftApprovals[id];
                    }
                } while (++toIndex != n);

                toAddressData.ownedLength = uint32(t.toOwnedLength = toIndex);
                fromAddressData.ownedLength = uint32(t.fromOwnedLength);
                break;
            }

            t.totalNFTSupply = uint256($.totalNFTSupply) + t.numNFTMints - t.numNFTBurns;
            $.totalNFTSupply = uint32(t.totalNFTSupply);

            Uint32Map storage oo = $.oo;
            t.packedLogs = _packedLogsMalloc(t.numNFTBurns + t.numNFTMints);

            t.burnedPoolTail = $.burnedPoolTail;
            if (t.numNFTBurns != 0) {
                _packedLogsSet(t.packedLogs, from, 1);
                bool addToBurnedPool = _addToBurnedPool(t.totalNFTSupply, $.totalSupply);
                Uint32Map storage fromOwned = $.owned[from];
                uint256 fromIndex = t.fromOwnedLength;
                fromAddressData.ownedLength = uint32(t.fromEnd = fromIndex - t.numNFTBurns);
                uint32 burnedPoolTail = t.burnedPoolTail;
                // Burn loop.
                do {
                    uint256 id = _get(fromOwned, --fromIndex);
                    _setOwnerAliasAndOwnedIndex(oo, id, 0, 0);
                    _packedLogsAppend(t.packedLogs, id);
                    if (_useExistsLookup()) _set($.exists, id, false);
                    if (addToBurnedPool) _set($.burnedPool, burnedPoolTail++, uint32(id));
                    if (_get($.mayHaveNFTApproval, id)) {
                        _set($.mayHaveNFTApproval, id, false);
                        delete $.nftApprovals[id];
                    }
                } while (fromIndex != t.fromEnd);

                if (addToBurnedPool) $.burnedPoolTail = (t.burnedPoolTail = burnedPoolTail);
            }

            if (t.numNFTMints != 0) {
                _packedLogsSet(t.packedLogs, to, 0);
                Uint32Map storage toOwned = $.owned[to];
                t.toAlias = _registerAndResolveAlias(toAddressData, to);
                uint256 idLimit = $.totalSupply / _unit();
                t.nextTokenId = _wrapNFTId($.nextTokenId, idLimit);
                uint256 toIndex = t.toOwnedLength;
                toAddressData.ownedLength = uint32(t.toEnd = toIndex + t.numNFTMints);
                uint32 burnedPoolHead = $.burnedPoolHead;
                // Mint loop.
                do {
                    uint256 id;
                    if (burnedPoolHead != t.burnedPoolTail) {
                        id = _get($.burnedPool, burnedPoolHead++);
                    } else {
                        id = t.nextTokenId;
                        while (_get(oo, _ownershipIndex(id)) != 0) {
                            id = _useExistsLookup()
                                ? _wrapNFTId(_findFirstUnset($.exists, id + 1, idLimit), idLimit)
                                : _wrapNFTId(id + 1, idLimit);
                        }
                        t.nextTokenId = _wrapNFTId(id + 1, idLimit);
                    }
                    if (_useExistsLookup()) _set($.exists, id, true);
                    _set(toOwned, toIndex, uint32(id));
                    _setOwnerAliasAndOwnedIndex(oo, id, t.toAlias, uint32(toIndex++));
                    _packedLogsAppend(t.packedLogs, id);
                } while (toIndex != t.toEnd);

                $.burnedPoolHead = burnedPoolHead;
                $.nextTokenId = uint32(t.nextTokenId);
            }

            if (t.directLogs != bytes32(0)) _directLogsSend(t.directLogs, $);
            if (t.packedLogs != bytes32(0)) _packedLogsSend(t.packedLogs, $);
        }
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Transfer} event.
            mstore(0x00, amount)
            // forgefmt: disable-next-item
            log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, shr(96, shl(96, from)), shr(96, shl(96, to)))
        }
        if (_useAfterNFTTransfers()) {
            uint256[] memory ids = _directLogsIds(t.directLogs);
            unchecked {
                _afterNFTTransfers(
                    _concat(
                        _filled(ids.length + t.numNFTBurns, from), _zeroAddresses(t.numNFTMints)
                    ),
                    _concat(
                        _concat(_filled(ids.length, to), _zeroAddresses(t.numNFTBurns)),
                        _filled(t.numNFTMints, to)
                    ),
                    _concat(ids, _packedLogsIds(t.packedLogs))
                );
            }
        }
    }

    /// @dev Transfers token `id` from `from` to `to`.
    /// Also emits an ERC721 {Transfer} event on the `mirrorERC721`.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - `msgSender` must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function _initiateTransferFromNFT(address from, address to, uint256 id, address msgSender)
        internal
        virtual
    {
        // Emit ERC721 {Transfer} event.
        // We do this before the `_transferFromNFT`, as `_transferFromNFT` may use
        // the `_afterNFTTransfers` hook, which may trigger more transfers.
        // This helps keeps the sequence of emitted events consistent.
        // Since `mirrorERC721` is a trusted contract, we can do this.
        bytes32 directLogs = _directLogsMalloc(1, from, to);
        _directLogsAppend(directLogs, id);
        _directLogsSend(directLogs, _getDN404Storage());

        _transferFromNFT(from, to, id, msgSender);
    }

    /// @dev Transfers token `id` from `from` to `to`.
    ///
    /// This function will be called when a ERC721 transfer is made on the mirror contract.
    ///
    /// Requirements:
    ///
    /// - Token `id` must exist.
    /// - `from` must be the owner of the token.
    /// - `to` cannot be the zero address.
    /// - `msgSender` must be the owner of the token, or be approved to manage the token.
    ///
    /// Emits a {Transfer} event.
    function _transferFromNFT(address from, address to, uint256 id, address msgSender)
        internal
        virtual
    {
        if (to == address(0)) revert TransferToZeroAddress();

        DN404Storage storage $ = _getDN404Storage();
        if ($.mirrorERC721 == address(0)) revert DNNotInitialized();

        Uint32Map storage oo = $.oo;

        if (from != $.aliasToAddress[_get(oo, _ownershipIndex(_restrictNFTId(id)))]) {
            revert TransferFromIncorrectOwner();
        }

        if (msgSender != from) {
            if (!_isApprovedForAll(from, msgSender)) {
                if (_getApproved(id) != msgSender) {
                    revert TransferCallerNotOwnerNorApproved();
                }
            }
        }

        AddressData storage fromAddressData = $.addressData[from];
        AddressData storage toAddressData = $.addressData[to];

        uint256 unit = _unit();
        mapping(address => Uint32Map) storage owned = $.owned;

        unchecked {
            uint256 fromBalance = fromAddressData.balance;
            if (unit > fromBalance) revert InsufficientBalance();
            fromAddressData.balance = uint96(fromBalance - unit);
            toAddressData.balance += uint96(unit);
        }
        if (_get($.mayHaveNFTApproval, id)) {
            _set($.mayHaveNFTApproval, id, false);
            delete $.nftApprovals[id];
        }
        unchecked {
            Uint32Map storage fromOwned = owned[from];
            uint32 updatedId = _get(fromOwned, --fromAddressData.ownedLength);
            uint32 i = _get(oo, _ownedIndex(id));
            _set(fromOwned, i, updatedId);
            _set(oo, _ownedIndex(updatedId), i);
        }
        unchecked {
            uint32 n = toAddressData.ownedLength++;
            _set(owned[to], n, uint32(id));
            _setOwnerAliasAndOwnedIndex(oo, id, _registerAndResolveAlias(toAddressData, to), n);
        }
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Transfer} event.
            mstore(0x00, unit)
            // forgefmt: disable-next-item
            log3(0x00, 0x20, _TRANSFER_EVENT_SIGNATURE, shr(96, shl(96, from)), shr(96, shl(96, to)))
        }
        if (_useAfterNFTTransfers()) {
            _afterNFTTransfers(_filled(1, from), _filled(1, to), _filled(1, id));
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                 INTERNAL APPROVE FUNCTIONS                 */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Sets `amount` as the allowance of `spender` over the tokens of `owner`.
    ///
    /// Emits a {Approval} event.
    function _approve(address owner, address spender, uint256 amount) internal virtual {
        if (_givePermit2DefaultInfiniteAllowance() && spender == _PERMIT2) {
            _getDN404Storage().addressData[owner].flags |= _ADDRESS_DATA_OVERRIDE_PERMIT2_FLAG;
        }
        _ref(_getDN404Storage().allowance, owner, spender).value = amount;
        /// @solidity memory-safe-assembly
        assembly {
            // Emit the {Approval} event.
            mstore(0x00, amount)
            // forgefmt: disable-next-item
            log3(0x00, 0x20, _APPROVAL_EVENT_SIGNATURE, shr(96, shl(96, owner)), shr(96, shl(96, spender)))
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                 DATA HITCHHIKING FUNCTIONS                 */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the auxiliary data for `owner`.
    /// Minting, transferring, burning the tokens of `owner` will not change the auxiliary data.
    /// Auxiliary data can be set for any address, even if it does not have any tokens.
    function _getAux(address owner) internal view virtual returns (uint88) {
        return _getDN404Storage().addressData[owner].aux;
    }

    /// @dev Set the auxiliary data for `owner` to `value`.
    /// Minting, transferring, burning the tokens of `owner` will not change the auxiliary data.
    /// Auxiliary data can be set for any address, even if it does not have any tokens.
    function _setAux(address owner, uint88 value) internal virtual {
        _getDN404Storage().addressData[owner].aux = value;
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                     SKIP NFT FUNCTIONS                     */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns true if minting and transferring ERC20s to `owner` will skip minting NFTs.
    /// Returns false otherwise.
    function getSkipNFT(address owner) public view virtual returns (bool result) {
        uint8 flags = _getDN404Storage().addressData[owner].flags;
        /// @solidity memory-safe-assembly
        assembly {
            result := iszero(iszero(and(flags, _ADDRESS_DATA_SKIP_NFT_FLAG)))
            if iszero(and(flags, _ADDRESS_DATA_SKIP_NFT_INITIALIZED_FLAG)) {
                result := iszero(iszero(extcodesize(owner)))
            }
        }
    }

    /// @dev Sets the caller's skipNFT flag to `skipNFT`. Returns true.
    ///
    /// Emits a {SkipNFTSet} event.
    function setSkipNFT(bool skipNFT) public virtual returns (bool) {
        _setSkipNFT(msg.sender, skipNFT);
        return true;
    }

    /// @dev Internal function to set account `owner` skipNFT flag to `state`
    ///
    /// Initializes account `owner` AddressData if it is not currently initialized.
    ///
    /// Emits a {SkipNFTSet} event.
    function _setSkipNFT(address owner, bool state) internal virtual {
        AddressData storage d = _getDN404Storage().addressData[owner];
        uint8 flags = d.flags;
        /// @solidity memory-safe-assembly
        assembly {
            let s := xor(iszero(and(flags, _ADDRESS_DATA_SKIP_NFT_FLAG)), iszero(state))
            flags := xor(mul(_ADDRESS_DATA_SKIP_NFT_FLAG, s), flags)
            flags := or(_ADDRESS_DATA_SKIP_NFT_INITIALIZED_FLAG, flags)
            mstore(0x00, iszero(iszero(state)))
            log2(0x00, 0x20, _SKIP_NFT_SET_EVENT_SIGNATURE, shr(96, shl(96, owner)))
        }
        d.flags = flags;
    }

    /// @dev Returns the `addressAlias` of account `to`.
    ///
    /// Assigns and registers the next alias if `to` alias was not previously registered.
    function _registerAndResolveAlias(AddressData storage toAddressData, address to)
        internal
        virtual
        returns (uint32 addressAlias)
    {
        DN404Storage storage $ = _getDN404Storage();
        addressAlias = toAddressData.addressAlias;
        if (addressAlias == uint256(0)) {
            unchecked {
                addressAlias = ++$.numAliases;
            }
            toAddressData.addressAlias = addressAlias;
            $.aliasToAddress[addressAlias] = to;
            if (addressAlias == uint256(0)) revert(); // Overflow.
        }
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                     MIRROR OPERATIONS                      */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns the address of the mirror NFT contract.
    function mirrorERC721() public view virtual returns (address) {
        return _getDN404Storage().mirrorERC721;
    }

    /// @dev Returns the total NFT supply.
    function _totalNFTSupply() internal view virtual returns (uint256) {
        return _getDN404Storage().totalNFTSupply;
    }

    /// @dev Returns `owner` NFT balance.
    function _balanceOfNFT(address owner) internal view virtual returns (uint256) {
        return _getDN404Storage().addressData[owner].ownedLength;
    }

    /// @dev Returns the owner of token `id`.
    /// Returns the zero address instead of reverting if the token does not exist.
    function _ownerAt(uint256 id) internal view virtual returns (address) {
        DN404Storage storage $ = _getDN404Storage();
        return $.aliasToAddress[_get($.oo, _ownershipIndex(_restrictNFTId(id)))];
    }

    /// @dev Returns the owner of token `id`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function _ownerOf(uint256 id) internal view virtual returns (address) {
        if (!_exists(id)) revert TokenDoesNotExist();
        return _ownerAt(id);
    }

    /// @dev Returns whether `operator` is approved to manage the NFT tokens of `owner`.
    function _isApprovedForAll(address owner, address operator)
        internal
        view
        virtual
        returns (bool)
    {
        return _ref(_getDN404Storage().operatorApprovals, owner, operator).value != 0;
    }

    /// @dev Returns if token `id` exists.
    function _exists(uint256 id) internal view virtual returns (bool) {
        return _ownerAt(id) != address(0);
    }

    /// @dev Returns the account approved to manage token `id`.
    ///
    /// Requirements:
    /// - Token `id` must exist.
    function _getApproved(uint256 id) internal view virtual returns (address) {
        if (!_exists(id)) revert TokenDoesNotExist();
        return _getDN404Storage().nftApprovals[id];
    }

    /// @dev Sets `spender` as the approved account to manage token `id`, using `msgSender`.
    ///
    /// Requirements:
    /// - `msgSender` must be the owner or an approved operator for the token owner.
    function _approveNFT(address spender, uint256 id, address msgSender)
        internal
        virtual
        returns (address owner)
    {
        DN404Storage storage $ = _getDN404Storage();

        owner = $.aliasToAddress[_get($.oo, _ownershipIndex(_restrictNFTId(id)))];

        if (msgSender != owner) {
            if (!_isApprovedForAll(owner, msgSender)) {
                revert ApprovalCallerNotOwnerNorApproved();
            }
        }

        $.nftApprovals[id] = spender;
        _set($.mayHaveNFTApproval, id, spender != address(0));
    }

    /// @dev Approve or remove the `operator` as an operator for `msgSender`,
    /// without authorization checks.
    function _setApprovalForAll(address operator, bool approved, address msgSender)
        internal
        virtual
    {
        // For efficiency, we won't check if `operator` isn't `address(0)` (practically a no-op).
        _ref(_getDN404Storage().operatorApprovals, msgSender, operator).value = _toUint(approved);
    }

    /// @dev Returns the NFT IDs of `owner` in the range `[begin..end)` (exclusive of `end`).
    /// `begin` and `end` are indices in the owner's token ID array, not the entire token range.
    /// Optimized for smaller bytecode size, as this function is intended for off-chain calling.
    function _ownedIds(address owner, uint256 begin, uint256 end)
        internal
        view
        virtual
        returns (uint256[] memory ids)
    {
        DN404Storage storage $ = _getDN404Storage();
        Uint32Map storage owned = $.owned[owner];
        end = _min($.addressData[owner].ownedLength, end);
        /// @solidity memory-safe-assembly
        assembly {
            ids := mload(0x40)
            let i := begin
            for {} lt(i, end) { i := add(i, 1) } {
                let s := add(shl(96, owned.slot), shr(3, i)) // Storage slot.
                let id := and(0xffffffff, shr(shl(5, and(i, 7)), sload(s)))
                mstore(add(add(ids, 0x20), shl(5, sub(i, begin))), id) // Append to.
            }
            mstore(ids, sub(i, begin)) // Store the length.
            mstore(0x40, add(add(ids, 0x20), shl(5, sub(i, begin)))) // Allocate memory.
        }
    }

    /// @dev Fallback modifier to dispatch calls from the mirror NFT contract
    /// to internal functions in this contract.
    modifier dn404Fallback() virtual {
        DN404Storage storage $ = _getDN404Storage();

        uint256 fnSelector = _calldataload(0x00) >> 224;

        // `transferFromNFT(address,address,uint256,address)`.
        if (fnSelector == 0xe5eb36c8) {
            if (msg.sender != $.mirrorERC721) revert SenderNotMirror();
            _transferFromNFT(
                address(uint160(_calldataload(0x04))), // `from`.
                address(uint160(_calldataload(0x24))), // `to`.
                _calldataload(0x44), // `id`.
                address(uint160(_calldataload(0x64))) // `msgSender`.
            );
            _return(1);
        }
        // `setApprovalForAllNFT(address,bool,address)`.
        if (fnSelector == 0xf6916ddd) {
            if (msg.sender != $.mirrorERC721) revert SenderNotMirror();
            _setApprovalForAll(
                address(uint160(_calldataload(0x04))), // `spender`.
                _calldataload(0x24) != 0, // `status`.
                address(uint160(_calldataload(0x44))) // `msgSender`.
            );
            _return(1);
        }
        // `isApprovedForAllNFT(address,address)`.
        if (fnSelector == 0x62fb246d) {
            bool result = _isApprovedForAll(
                address(uint160(_calldataload(0x04))), // `owner`.
                address(uint160(_calldataload(0x24))) // `operator`.
            );
            _return(_toUint(result));
        }
        // `ownerOfNFT(uint256)`.
        if (fnSelector == 0x2d8a746e) {
            _return(uint160(_ownerOf(_calldataload(0x04))));
        }
        // `ownerAtNFT(uint256)`.
        if (fnSelector == 0xc016aa52) {
            _return(uint160(_ownerAt(_calldataload(0x04))));
        }
        // `approveNFT(address,uint256,address)`.
        if (fnSelector == 0xd10b6e0c) {
            if (msg.sender != $.mirrorERC721) revert SenderNotMirror();
            address owner = _approveNFT(
                address(uint160(_calldataload(0x04))), // `spender`.
                _calldataload(0x24), // `id`.
                address(uint160(_calldataload(0x44))) // `msgSender`.
            );
            _return(uint160(owner));
        }
        // `getApprovedNFT(uint256)`.
        if (fnSelector == 0x27ef5495) {
            _return(uint160(_getApproved(_calldataload(0x04))));
        }
        // `balanceOfNFT(address)`.
        if (fnSelector == 0xf5b100ea) {
            _return(_balanceOfNFT(address(uint160(_calldataload(0x04)))));
        }
        // `totalNFTSupply()`.
        if (fnSelector == 0xe2c79281) {
            _return(_totalNFTSupply());
        }
        // `tokenURINFT(uint256)`.
        if (fnSelector == 0xcb30b460) {
            /// @solidity memory-safe-assembly
            assembly {
                mstore(0x40, add(mload(0x40), 0x20))
            }
            string memory uri = _tokenURI(_calldataload(0x04));
            /// @solidity memory-safe-assembly
            assembly {
                // Memory safe, as we've advanced the free memory pointer by a word.
                let o := sub(uri, 0x20) // Start of the returndata.
                let z := add(mload(uri), 0x40) // Unpadded length of returndata.
                mstore(add(o, z), 0) // Zeroize the word after the end of the string.
                mstore(o, 0x20) // Store the offset of `uri`.
                return(o, and(not(0x1f), add(0x1f, z)))
            }
        }
        // `implementsDN404()`.
        if (fnSelector == 0xb7a94eb8) {
            _return(1);
        }
        _;
    }

    /// @dev Fallback function for calls from mirror NFT contract.
    /// Override this if you need to implement your custom
    /// fallback with utilities like Solady's `LibZip.cdFallback()`.
    /// And always remember to always wrap the fallback with `dn404Fallback`.
    fallback() external payable virtual dn404Fallback {
        revert FnSelectorNotRecognized(); // Not mandatory. Just for quality of life.
    }

    /// @dev This is to silence the compiler warning.
    /// Override and remove the revert if you want your contract to receive ETH via receive.
    receive() external payable virtual {
        if (msg.value != 0) revert();
    }

    /*«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-«-*/
    /*                 INTERNAL / PRIVATE HELPERS                 */
    /*-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»-»*/

    /// @dev Returns `(i - _toUint(_useOneIndexed())) << 1`.
    function _ownershipIndex(uint256 i) internal pure returns (uint256) {
        unchecked {
            return (i - _toUint(_useOneIndexed())) << 1;
        }
    }

    /// @dev Returns `((i - _toUint(_useOneIndexed())) << 1) + 1`.
    function _ownedIndex(uint256 i) internal pure returns (uint256) {
        unchecked {
            return ((i - _toUint(_useOneIndexed())) << 1) + 1;
        }
    }

    /// @dev Returns the uint32 value at `index` in `map`.
    function _get(Uint32Map storage map, uint256 index) internal view returns (uint32 result) {
        /// @solidity memory-safe-assembly
        assembly {
            let s := add(shl(96, map.slot), shr(3, index)) // Storage slot.
            result := and(0xffffffff, shr(shl(5, and(index, 7)), sload(s)))
        }
    }

    /// @dev Updates the uint32 value at `index` in `map`.
    function _set(Uint32Map storage map, uint256 index, uint32 value) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let s := add(shl(96, map.slot), shr(3, index)) // Storage slot.
            let o := shl(5, and(index, 7)) // Storage slot offset (bits).
            let v := sload(s) // Storage slot value.
            sstore(s, xor(v, shl(o, and(0xffffffff, xor(value, shr(o, v))))))
        }
    }

    /// @dev Sets the owner alias and the owned index together.
    function _setOwnerAliasAndOwnedIndex(
        Uint32Map storage map,
        uint256 id,
        uint32 ownership,
        uint32 ownedIndex
    ) internal {
        uint256 t = _toUint(_useOneIndexed());
        /// @solidity memory-safe-assembly
        assembly {
            let i := sub(id, t) // Index of the uint64 combined value.
            let s := add(shl(96, map.slot), shr(2, i)) // Storage slot.
            let v := sload(s) // Storage slot value.
            let o := shl(6, and(i, 3)) // Storage slot offset (bits).
            let combined := or(shl(32, ownedIndex), and(0xffffffff, ownership))
            sstore(s, xor(v, shl(o, and(0xffffffffffffffff, xor(shr(o, v), combined)))))
        }
    }

    /// @dev Returns the boolean value of the bit at `index` in `bitmap`.
    function _get(Bitmap storage bitmap, uint256 index) internal view returns (bool result) {
        /// @solidity memory-safe-assembly
        assembly {
            let s := add(shl(96, bitmap.slot), shr(8, index)) // Storage slot.
            result := and(1, shr(and(0xff, index), sload(s)))
        }
    }

    /// @dev Updates the bit at `index` in `bitmap` to `value`.
    function _set(Bitmap storage bitmap, uint256 index, bool value) internal {
        /// @solidity memory-safe-assembly
        assembly {
            let s := add(shl(96, bitmap.slot), shr(8, index)) // Storage slot.
            let o := and(0xff, index) // Storage slot offset (bits).
            sstore(s, or(and(sload(s), not(shl(o, 1))), shl(o, iszero(iszero(value)))))
        }
    }

    /// @dev Returns the index of the least significant unset bit in `[begin..upTo]`.
    /// If no unset bit is found, returns `type(uint256).max`.
    function _findFirstUnset(Bitmap storage bitmap, uint256 begin, uint256 upTo)
        internal
        view
        returns (uint256 unsetBitIndex)
    {
        /// @solidity memory-safe-assembly
        assembly {
            unsetBitIndex := not(0) // Initialize to `type(uint256).max`.
            let s := shl(96, bitmap.slot) // Storage offset of the bitmap.
            let bucket := add(s, shr(8, begin))
            let negBits := shl(and(0xff, begin), shr(and(0xff, begin), not(sload(bucket))))
            if iszero(negBits) {
                let lastBucket := add(s, shr(8, upTo))
                for {} 1 {} {
                    bucket := add(bucket, 1)
                    negBits := not(sload(bucket))
                    if or(negBits, gt(bucket, lastBucket)) { break }
                }
                if gt(bucket, lastBucket) {
                    negBits := shr(and(0xff, not(upTo)), shl(and(0xff, not(upTo)), negBits))
                }
            }
            if negBits {
                // Find-first-set routine.
                // From: https://github.com/vectorized/solady/blob/main/src/utils/LibBit.sol
                let b := and(negBits, add(not(negBits), 1)) // Isolate the least significant bit.
                // For the upper 3 bits of the result, use a De Bruijn-like lookup.
                // Credit to adhusson: https://blog.adhusson.com/cheap-find-first-set-evm/
                // forgefmt: disable-next-item
                let r := shl(5, shr(252, shl(shl(2, shr(250, mul(b,
                    0x2aaaaaaaba69a69a6db6db6db2cb2cb2ce739ce73def7bdeffffffff))),
                    0x1412563212c14164235266736f7425221143267a45243675267677)))
                // For the lower 5 bits of the result, use a De Bruijn lookup.
                // forgefmt: disable-next-item
                r := or(r, byte(and(div(0xd76453e0, shr(r, b)), 0x1f),
                    0x001f0d1e100c1d070f090b19131c1706010e11080a1a141802121b1503160405))
                r := or(shl(8, sub(bucket, s)), r)
                unsetBitIndex := or(r, sub(0, or(gt(r, upTo), lt(r, begin))))
            }
        }
    }

    /// @dev Returns a storage reference to the value at (`a0`, `a1`) in `map`.
    function _ref(AddressPairToUint256RefMap storage map, address a0, address a1)
        internal
        pure
        returns (Uint256Ref storage ref)
    {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x28, a1)
            mstore(0x14, a0)
            mstore(0x00, map.slot)
            ref.slot := keccak256(0x00, 0x48)
            // Clear the part of the free memory pointer that was overwritten.
            mstore(0x28, 0x00)
        }
    }

    /// @dev Wraps the NFT ID.
    function _wrapNFTId(uint256 id, uint256 idLimit) internal pure returns (uint256 result) {
        result = _toUint(_useOneIndexed());
        /// @solidity memory-safe-assembly
        assembly {
            result :=
                or(
                    mul(or(mul(iszero(gt(id, idLimit)), id), gt(id, idLimit)), result),
                    mul(mul(lt(id, idLimit), id), iszero(result))
                )
        }
    }

    /// @dev Returns `id > type(uint32).max ? 0 : id`.
    function _restrictNFTId(uint256 id) internal pure returns (uint256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mul(id, lt(id, 0x100000000))
        }
    }

    /// @dev Returns whether `amount` is an invalid `totalSupply`.
    function _totalSupplyOverflows(uint256 amount) internal view returns (bool result) {
        uint256 unit = _unit();
        /// @solidity memory-safe-assembly
        assembly {
            result := iszero(iszero(or(shr(96, amount), lt(0xfffffffe, div(amount, unit)))))
        }
    }

    /// @dev Returns `max(0, x - y)`.
    function _zeroFloorSub(uint256 x, uint256 y) internal pure returns (uint256 z) {
        /// @solidity memory-safe-assembly
        assembly {
            z := mul(gt(x, y), sub(x, y))
        }
    }

    /// @dev Returns `x < y ? x : y`.
    function _min(uint256 x, uint256 y) internal pure returns (uint256 z) {
        /// @solidity memory-safe-assembly
        assembly {
            z := xor(x, mul(xor(x, y), lt(y, x)))
        }
    }

    /// @dev Returns `b ? 1 : 0`.
    function _toUint(bool b) internal pure returns (uint256 result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := iszero(iszero(b))
        }
    }

    /// @dev Initiates memory allocation for direct logs with `n` log items.
    function _directLogsMalloc(uint256 n, address from, address to)
        private
        pure
        returns (bytes32 p)
    {
        /// @solidity memory-safe-assembly
        assembly {
            // `p`'s layout:
            //    uint256 offset;
            //    uint256[] logs;
            p := mload(0x40)
            let m := add(p, 0x40)
            mstore(m, 0x144027d3) // `logDirectTransfer(address,address,uint256[])`.
            mstore(add(m, 0x20), shr(96, shl(96, from)))
            mstore(add(m, 0x40), shr(96, shl(96, to)))
            mstore(add(m, 0x60), 0x60) // Offset of `logs` in the calldata to send.
            // Skip 4 words: `fnSelector`, `from`, `to`, `calldataLogsOffset`.
            let logs := add(0x80, m)
            mstore(logs, n) // Store the length.
            let offset := add(0x20, logs) // Skip the word for `p.logs.length`.
            mstore(0x40, add(offset, shl(5, n))) // Allocate memory.
            mstore(add(0x20, p), logs) // Set `p.logs`.
            mstore(p, offset) // Set `p.offset`.
        }
    }

    /// @dev Adds a direct log item to `p` with token `id`.
    function _directLogsAppend(bytes32 p, uint256 id) private pure {
        /// @solidity memory-safe-assembly
        assembly {
            let offset := mload(p)
            mstore(offset, id)
            mstore(p, add(offset, 0x20))
        }
    }

    /// @dev Calls the `mirror` NFT contract to emit {Transfer} events for packed logs `p`.
    function _directLogsSend(bytes32 p, DN404Storage storage $) private {
        address mirror = $.mirrorERC721;
        /// @solidity memory-safe-assembly
        assembly {
            let logs := mload(add(p, 0x20))
            let n := add(0x84, shl(5, mload(logs))) // Length of calldata to send.
            let o := sub(logs, 0x80) // Start of calldata to send.
            if iszero(and(eq(mload(o), 1), call(gas(), mirror, 0, add(o, 0x1c), n, o, 0x20))) {
                revert(o, 0x00)
            }
        }
    }

    /// @dev Returns the token IDs of the direct logs.
    function _directLogsIds(bytes32 p) private pure returns (uint256[] memory ids) {
        /// @solidity memory-safe-assembly
        assembly {
            if p { ids := mload(add(p, 0x20)) }
        }
    }

    /// @dev Initiates memory allocation for packed logs with `n` log items.
    function _packedLogsMalloc(uint256 n) private pure returns (bytes32 p) {
        /// @solidity memory-safe-assembly
        assembly {
            // `p`'s layout:
            //     uint256 offset;
            //     uint256 addressAndBit;
            //     uint256[] logs;
            p := mload(0x40)
            let logs := add(p, 0xa0)
            mstore(logs, n) // Store the length.
            let offset := add(0x20, logs) // Skip the word for `p.logs.length`.
            mstore(0x40, add(offset, shl(5, n))) // Allocate memory.
            mstore(add(0x40, p), logs) // Set `p.logs`.
            mstore(p, offset) // Set `p.offset`.
        }
    }

    /// @dev Set the current address and the burn bit.
    function _packedLogsSet(bytes32 p, address a, uint256 burnBit) private pure {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(add(p, 0x20), or(shl(96, a), burnBit)) // Set `p.addressAndBit`.
        }
    }

    /// @dev Adds a packed log item to `p` with token `id`.
    function _packedLogsAppend(bytes32 p, uint256 id) private pure {
        /// @solidity memory-safe-assembly
        assembly {
            let offset := mload(p)
            mstore(offset, or(mload(add(p, 0x20)), shl(8, id))) // `p.addressAndBit | (id << 8)`.
            mstore(p, add(offset, 0x20))
        }
    }

    /// @dev Calls the `mirror` NFT contract to emit {Transfer} events for packed logs `p`.
    function _packedLogsSend(bytes32 p, DN404Storage storage $) private {
        address mirror = $.mirrorERC721;
        /// @solidity memory-safe-assembly
        assembly {
            let logs := mload(add(p, 0x40))
            let o := sub(logs, 0x40) // Start of calldata to send.
            mstore(o, 0x263c69d6) // `logTransfer(uint256[])`.
            mstore(add(o, 0x20), 0x20) // Offset of `logs` in the calldata to send.
            let n := add(0x44, shl(5, mload(logs))) // Length of calldata to send.
            if iszero(and(eq(mload(o), 1), call(gas(), mirror, 0, add(o, 0x1c), n, o, 0x20))) {
                revert(o, 0x00)
            }
        }
    }

    /// @dev Returns the token IDs of the packed logs (destructively).
    function _packedLogsIds(bytes32 p) private pure returns (uint256[] memory ids) {
        /// @solidity memory-safe-assembly
        assembly {
            if p {
                ids := mload(add(p, 0x40))
                let o := add(ids, 0x20)
                let end := add(o, shl(5, mload(ids)))
                for {} iszero(eq(o, end)) { o := add(o, 0x20) } {
                    mstore(o, shr(168, shl(160, mload(o))))
                }
            }
        }
    }

    /// @dev Returns an array of zero addresses.
    function _zeroAddresses(uint256 n) private pure returns (address[] memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            mstore(0x40, add(add(result, 0x20), shl(5, n)))
            mstore(result, n)
            codecopy(add(result, 0x20), codesize(), shl(5, n))
        }
    }

    /// @dev Returns an array each set to `value`.
    function _filled(uint256 n, uint256 value) private pure returns (uint256[] memory result) {
        /// @solidity memory-safe-assembly
        assembly {
            result := mload(0x40)
            let o := add(result, 0x20)
            let end := add(o, shl(5, n))
            mstore(0x40, end)
            mstore(result, n)
            for {} iszero(eq(o, end)) { o := add(o, 0x20) } { mstore(o, value) }
        }
    }

    /// @dev Returns an array each set to `value`.
    function _filled(uint256 n, address value) private pure returns (address[] memory result) {
        result = _toAddresses(_filled(n, uint160(value)));
    }

    /// @dev Concatenates the arrays.
    function _concat(uint256[] memory a, uint256[] memory b)
        private
        view
        returns (uint256[] memory result)
    {
        uint256 aN = a.length;
        uint256 bN = b.length;
        if (aN == uint256(0)) return b;
        if (bN == uint256(0)) return a;
        /// @solidity memory-safe-assembly
        assembly {
            let n := add(aN, bN)
            if n {
                result := mload(0x40)
                mstore(result, n)
                let o := add(result, 0x20)
                mstore(0x40, add(o, shl(5, n)))
                let aL := shl(5, aN)
                pop(staticcall(gas(), 4, add(a, 0x20), aL, o, aL))
                pop(staticcall(gas(), 4, add(b, 0x20), shl(5, bN), add(o, aL), shl(5, bN)))
            }
        }
    }

    /// @dev Concatenates the arrays.
    function _concat(address[] memory a, address[] memory b)
        private
        view
        returns (address[] memory result)
    {
        result = _toAddresses(_concat(_toUints(a), _toUints(b)));
    }

    /// @dev Reinterpret cast to an uint array.
    function _toUints(address[] memory a) private pure returns (uint256[] memory casted) {
        /// @solidity memory-safe-assembly
        assembly {
            casted := a
        }
    }

    /// @dev Reinterpret cast to an address array.
    function _toAddresses(uint256[] memory a) private pure returns (address[] memory casted) {
        /// @solidity memory-safe-assembly
        assembly {
            casted := a
        }
    }

    /// @dev Struct of temporary variables for transfers.
    struct _DNTransferTemps {
        uint256 numNFTBurns;
        uint256 numNFTMints;
        uint256 fromOwnedLength;
        uint256 toOwnedLength;
        uint256 totalNFTSupply;
        uint256 fromEnd;
        uint256 toEnd;
        uint32 toAlias;
        uint256 nextTokenId;
        uint32 burnedPoolTail;
        bytes32 directLogs;
        bytes32 packedLogs;
    }

    /// @dev Struct of temporary variables for mints.
    struct _DNMintTemps {
        uint256 nextTokenId;
        uint32 burnedPoolTail;
        uint256 toEnd;
        uint32 toAlias;
        uint256 numNFTMints;
        bytes32 packedLogs;
    }

    /// @dev Struct of temporary variables for burns.
    struct _DNBurnTemps {
        uint256 fromBalance;
        uint256 totalSupply;
        uint256 numNFTBurns;
        bytes32 packedLogs;
    }

    /// @dev Returns the calldata value at `offset`.
    function _calldataload(uint256 offset) private pure returns (uint256 value) {
        /// @solidity memory-safe-assembly
        assembly {
            value := calldataload(offset)
        }
    }

    /// @dev Executes a return opcode to return `x` and end the current call frame.
    function _return(uint256 x) private pure {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, x)
            return(0x00, 0x20)
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

// File: @openzeppelin/contracts/access/Ownable.sol


// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

pragma solidity ^0.8.20;


/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// File: @openzeppelin/contracts/utils/math/Math.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/math/Math.sol)

pragma solidity ^0.8.20;

/**
 * @dev Standard math utilities missing in the Solidity language.
 */
library Math {
    /**
     * @dev Muldiv operation overflow.
     */
    error MathOverflowedMulDiv();

    enum Rounding {
        Floor, // Toward negative infinity
        Ceil, // Toward positive infinity
        Trunc, // Toward zero
        Expand // Away from zero
    }

    /**
     * @dev Returns the addition of two unsigned integers, with an overflow flag.
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
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an overflow flag.
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
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a division by zero flag.
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool, uint256) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Returns the largest of two numbers.
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    /**
     * @dev Returns the smallest of two numbers.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    /**
     * @dev Returns the average of two numbers. The result is rounded towards
     * zero.
     */
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        // (a + b) / 2 can overflow.
        return (a & b) + (a ^ b) / 2;
    }

    /**
     * @dev Returns the ceiling of the division of two numbers.
     *
     * This differs from standard division with `/` in that it rounds towards infinity instead
     * of rounding towards zero.
     */
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) {
            // Guarantee the same behavior as in a regular Solidity division.
            return a / b;
        }

        // (a + b - 1) / b can overflow on addition, so we distribute.
        return a == 0 ? 0 : (a - 1) / b + 1;
    }

    /**
     * @notice Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or
     * denominator == 0.
     * @dev Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv) with further edits by
     * Uniswap Labs also under MIT license.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2^256 and mod 2^256 - 1, then use
            // use the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
            // variables such that product = prod1 * 2^256 + prod0.
            uint256 prod0 = x * y; // Least significant 256 bits of the product
            uint256 prod1; // Most significant 256 bits of the product
            assembly {
                let mm := mulmod(x, y, not(0))
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            // Handle non-overflow cases, 256 by 256 division.
            if (prod1 == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return prod0 / denominator;
            }

            // Make sure the result is less than 2^256. Also prevents denominator == 0.
            if (denominator <= prod1) {
                revert MathOverflowedMulDiv();
            }

            ///////////////////////////////////////////////
            // 512 by 256 division.
            ///////////////////////////////////////////////

            // Make division exact by subtracting the remainder from [prod1 prod0].
            uint256 remainder;
            assembly {
                // Compute remainder using mulmod.
                remainder := mulmod(x, y, denominator)

                // Subtract 256 bit number from 512 bit number.
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator.
            // Always >= 1. See https://cs.stackexchange.com/q/138556/92363.

            uint256 twos = denominator & (0 - denominator);
            assembly {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [prod1 prod0] by twos.
                prod0 := div(prod0, twos)

                // Flip twos such that it is 2^256 / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from prod1 into prod0.
            prod0 |= prod1 * twos;

            // Invert denominator mod 2^256. Now that denominator is an odd number, it has an inverse modulo 2^256 such
            // that denominator * inv = 1 mod 2^256. Compute the inverse by starting with a seed that is correct for
            // four bits. That is, denominator * inv = 1 mod 2^4.
            uint256 inverse = (3 * denominator) ^ 2;

            // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also
            // works in modular arithmetic, doubling the correct bits in each step.
            inverse *= 2 - denominator * inverse; // inverse mod 2^8
            inverse *= 2 - denominator * inverse; // inverse mod 2^16
            inverse *= 2 - denominator * inverse; // inverse mod 2^32
            inverse *= 2 - denominator * inverse; // inverse mod 2^64
            inverse *= 2 - denominator * inverse; // inverse mod 2^128
            inverse *= 2 - denominator * inverse; // inverse mod 2^256

            // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
            // This will give us the correct result modulo 2^256. Since the preconditions guarantee that the outcome is
            // less than 2^256, this is the final result. We don't need to compute the high bits of the result and prod1
            // is no longer required.
            result = prod0 * inverse;
            return result;
        }
    }

    /**
     * @notice Calculates x * y / denominator with full precision, following the selected rounding direction.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator, Rounding rounding) internal pure returns (uint256) {
        uint256 result = mulDiv(x, y, denominator);
        if (unsignedRoundsUp(rounding) && mulmod(x, y, denominator) > 0) {
            result += 1;
        }
        return result;
    }

    /**
     * @dev Returns the square root of a number. If the number is not a perfect square, the value is rounded
     * towards zero.
     *
     * Inspired by Henry S. Warren, Jr.'s "Hacker's Delight" (Chapter 11).
     */
    function sqrt(uint256 a) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }

        // For our first guess, we get the biggest power of 2 which is smaller than the square root of the target.
        //
        // We know that the "msb" (most significant bit) of our target number `a` is a power of 2 such that we have
        // `msb(a) <= a < 2*msb(a)`. This value can be written `msb(a)=2**k` with `k=log2(a)`.
        //
        // This can be rewritten `2**log2(a) <= a < 2**(log2(a) + 1)`
        // → `sqrt(2**k) <= sqrt(a) < sqrt(2**(k+1))`
        // → `2**(k/2) <= sqrt(a) < 2**((k+1)/2) <= 2**(k/2 + 1)`
        //
        // Consequently, `2**(log2(a) / 2)` is a good first approximation of `sqrt(a)` with at least 1 correct bit.
        uint256 result = 1 << (log2(a) >> 1);

        // At this point `result` is an estimation with one bit of precision. We know the true value is a uint128,
        // since it is the square root of a uint256. Newton's method converges quadratically (precision doubles at
        // every iteration). We thus need at most 7 iteration to turn our partial result with one bit of precision
        // into the expected uint128 result.
        unchecked {
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            return min(result, a / result);
        }
    }

    /**
     * @notice Calculates sqrt(a), following the selected rounding direction.
     */
    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + (unsignedRoundsUp(rounding) && result * result < a ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 2 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 128;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 64;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 32;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 16;
            }
            if (value >> 8 > 0) {
                value >>= 8;
                result += 8;
            }
            if (value >> 4 > 0) {
                value >>= 4;
                result += 4;
            }
            if (value >> 2 > 0) {
                value >>= 2;
                result += 2;
            }
            if (value >> 1 > 0) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 2, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log2(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log2(value);
            return result + (unsignedRoundsUp(rounding) && 1 << result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 10 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log10(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >= 10 ** 64) {
                value /= 10 ** 64;
                result += 64;
            }
            if (value >= 10 ** 32) {
                value /= 10 ** 32;
                result += 32;
            }
            if (value >= 10 ** 16) {
                value /= 10 ** 16;
                result += 16;
            }
            if (value >= 10 ** 8) {
                value /= 10 ** 8;
                result += 8;
            }
            if (value >= 10 ** 4) {
                value /= 10 ** 4;
                result += 4;
            }
            if (value >= 10 ** 2) {
                value /= 10 ** 2;
                result += 2;
            }
            if (value >= 10 ** 1) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 10, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log10(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log10(value);
            return result + (unsignedRoundsUp(rounding) && 10 ** result < value ? 1 : 0);
        }
    }

    /**
     * @dev Return the log in base 256 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     *
     * Adding one to the result gives the number of pairs of hex symbols needed to represent `value` as a hex string.
     */
    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 16;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 8;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 4;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 2;
            }
            if (value >> 8 > 0) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 256, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log256(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log256(value);
            return result + (unsignedRoundsUp(rounding) && 1 << (result << 3) < value ? 1 : 0);
        }
    }

    /**
     * @dev Returns whether a provided rounding mode is considered rounding up for unsigned integers.
     */
    function unsignedRoundsUp(Rounding rounding) internal pure returns (bool) {
        return uint8(rounding) % 2 == 1;
    }
}

// File: @openzeppelin/contracts/utils/math/SignedMath.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/math/SignedMath.sol)

pragma solidity ^0.8.20;

/**
 * @dev Standard signed math utilities missing in the Solidity language.
 */
library SignedMath {
    /**
     * @dev Returns the largest of two signed numbers.
     */
    function max(int256 a, int256 b) internal pure returns (int256) {
        return a > b ? a : b;
    }

    /**
     * @dev Returns the smallest of two signed numbers.
     */
    function min(int256 a, int256 b) internal pure returns (int256) {
        return a < b ? a : b;
    }

    /**
     * @dev Returns the average of two signed numbers without overflow.
     * The result is rounded towards zero.
     */
    function average(int256 a, int256 b) internal pure returns (int256) {
        // Formula from the book "Hacker's Delight"
        int256 x = (a & b) + ((a ^ b) >> 1);
        return x + (int256(uint256(x) >> 255) & (a ^ b));
    }

    /**
     * @dev Returns the absolute unsigned value of a signed value.
     */
    function abs(int256 n) internal pure returns (uint256) {
        unchecked {
            // must be unchecked in order to support `n = type(int256).min`
            return uint256(n >= 0 ? n : -n);
        }
    }
}

// File: @openzeppelin/contracts/utils/Strings.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/Strings.sol)

pragma solidity ^0.8.20;



/**
 * @dev String operations.
 */
library Strings {
    bytes16 private constant HEX_DIGITS = "0123456789abcdef";
    uint8 private constant ADDRESS_LENGTH = 20;

    /**
     * @dev The `value` string doesn't fit in the specified `length`.
     */
    error StringsInsufficientHexLength(uint256 value, uint256 length);

    /**
     * @dev Converts a `uint256` to its ASCII `string` decimal representation.
     */
    function toString(uint256 value) internal pure returns (string memory) {
        unchecked {
            uint256 length = Math.log10(value) + 1;
            string memory buffer = new string(length);
            uint256 ptr;
            /// @solidity memory-safe-assembly
            assembly {
                ptr := add(buffer, add(32, length))
            }
            while (true) {
                ptr--;
                /// @solidity memory-safe-assembly
                assembly {
                    mstore8(ptr, byte(mod(value, 10), HEX_DIGITS))
                }
                value /= 10;
                if (value == 0) break;
            }
            return buffer;
        }
    }

    /**
     * @dev Converts a `int256` to its ASCII `string` decimal representation.
     */
    function toStringSigned(int256 value) internal pure returns (string memory) {
        return string.concat(value < 0 ? "-" : "", toString(SignedMath.abs(value)));
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation.
     */
    function toHexString(uint256 value) internal pure returns (string memory) {
        unchecked {
            return toHexString(value, Math.log256(value) + 1);
        }
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation with fixed length.
     */
    function toHexString(uint256 value, uint256 length) internal pure returns (string memory) {
        uint256 localValue = value;
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; --i) {
            buffer[i] = HEX_DIGITS[localValue & 0xf];
            localValue >>= 4;
        }
        if (localValue != 0) {
            revert StringsInsufficientHexLength(value, length);
        }
        return string(buffer);
    }

    /**
     * @dev Converts an `address` with fixed length of 20 bytes to its not checksummed ASCII `string` hexadecimal
     * representation.
     */
    function toHexString(address addr) internal pure returns (string memory) {
        return toHexString(uint256(uint160(addr)), ADDRESS_LENGTH);
    }

    /**
     * @dev Returns true if the two strings are equal.
     */
    function equal(string memory a, string memory b) internal pure returns (bool) {
        return bytes(a).length == bytes(b).length && keccak256(bytes(a)) == keccak256(bytes(b));
    }
}

// File: @openzeppelin/contracts/utils/cryptography/MerkleProof.sol


// OpenZeppelin Contracts (last updated v5.0.0) (utils/cryptography/MerkleProof.sol)

pragma solidity ^0.8.20;

/**
 * @dev These functions deal with verification of Merkle Tree proofs.
 *
 * The tree and the proofs can be generated using our
 * https://github.com/OpenZeppelin/merkle-tree[JavaScript library].
 * You will find a quickstart guide in the readme.
 *
 * WARNING: You should avoid using leaf values that are 64 bytes long prior to
 * hashing, or use a hash function other than keccak256 for hashing leaves.
 * This is because the concatenation of a sorted pair of internal nodes in
 * the Merkle tree could be reinterpreted as a leaf value.
 * OpenZeppelin's JavaScript library generates Merkle trees that are safe
 * against this attack out of the box.
 */
library MerkleProof {
    /**
     *@dev The multiproof provided is not valid.
     */
    error MerkleProofInvalidMultiproof();

    /**
     * @dev Returns true if a `leaf` can be proved to be a part of a Merkle tree
     * defined by `root`. For this, a `proof` must be provided, containing
     * sibling hashes on the branch from the leaf to the root of the tree. Each
     * pair of leaves and each pair of pre-images are assumed to be sorted.
     */
    function verify(bytes32[] memory proof, bytes32 root, bytes32 leaf) internal pure returns (bool) {
        return processProof(proof, leaf) == root;
    }

    /**
     * @dev Calldata version of {verify}
     */
    function verifyCalldata(bytes32[] calldata proof, bytes32 root, bytes32 leaf) internal pure returns (bool) {
        return processProofCalldata(proof, leaf) == root;
    }

    /**
     * @dev Returns the rebuilt hash obtained by traversing a Merkle tree up
     * from `leaf` using `proof`. A `proof` is valid if and only if the rebuilt
     * hash matches the root of the tree. When processing the proof, the pairs
     * of leafs & pre-images are assumed to be sorted.
     */
    function processProof(bytes32[] memory proof, bytes32 leaf) internal pure returns (bytes32) {
        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            computedHash = _hashPair(computedHash, proof[i]);
        }
        return computedHash;
    }

    /**
     * @dev Calldata version of {processProof}
     */
    function processProofCalldata(bytes32[] calldata proof, bytes32 leaf) internal pure returns (bytes32) {
        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            computedHash = _hashPair(computedHash, proof[i]);
        }
        return computedHash;
    }

    /**
     * @dev Returns true if the `leaves` can be simultaneously proven to be a part of a Merkle tree defined by
     * `root`, according to `proof` and `proofFlags` as described in {processMultiProof}.
     *
     * CAUTION: Not all Merkle trees admit multiproofs. See {processMultiProof} for details.
     */
    function multiProofVerify(
        bytes32[] memory proof,
        bool[] memory proofFlags,
        bytes32 root,
        bytes32[] memory leaves
    ) internal pure returns (bool) {
        return processMultiProof(proof, proofFlags, leaves) == root;
    }

    /**
     * @dev Calldata version of {multiProofVerify}
     *
     * CAUTION: Not all Merkle trees admit multiproofs. See {processMultiProof} for details.
     */
    function multiProofVerifyCalldata(
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32 root,
        bytes32[] memory leaves
    ) internal pure returns (bool) {
        return processMultiProofCalldata(proof, proofFlags, leaves) == root;
    }

    /**
     * @dev Returns the root of a tree reconstructed from `leaves` and sibling nodes in `proof`. The reconstruction
     * proceeds by incrementally reconstructing all inner nodes by combining a leaf/inner node with either another
     * leaf/inner node or a proof sibling node, depending on whether each `proofFlags` item is true or false
     * respectively.
     *
     * CAUTION: Not all Merkle trees admit multiproofs. To use multiproofs, it is sufficient to ensure that: 1) the tree
     * is complete (but not necessarily perfect), 2) the leaves to be proven are in the opposite order they are in the
     * tree (i.e., as seen from right to left starting at the deepest layer and continuing at the next layer).
     */
    function processMultiProof(
        bytes32[] memory proof,
        bool[] memory proofFlags,
        bytes32[] memory leaves
    ) internal pure returns (bytes32 merkleRoot) {
        // This function rebuilds the root hash by traversing the tree up from the leaves. The root is rebuilt by
        // consuming and producing values on a queue. The queue starts with the `leaves` array, then goes onto the
        // `hashes` array. At the end of the process, the last hash in the `hashes` array should contain the root of
        // the Merkle tree.
        uint256 leavesLen = leaves.length;
        uint256 proofLen = proof.length;
        uint256 totalHashes = proofFlags.length;

        // Check proof validity.
        if (leavesLen + proofLen != totalHashes + 1) {
            revert MerkleProofInvalidMultiproof();
        }

        // The xxxPos values are "pointers" to the next value to consume in each array. All accesses are done using
        // `xxx[xxxPos++]`, which return the current value and increment the pointer, thus mimicking a queue's "pop".
        bytes32[] memory hashes = new bytes32[](totalHashes);
        uint256 leafPos = 0;
        uint256 hashPos = 0;
        uint256 proofPos = 0;
        // At each step, we compute the next hash using two values:
        // - a value from the "main queue". If not all leaves have been consumed, we get the next leaf, otherwise we
        //   get the next hash.
        // - depending on the flag, either another value from the "main queue" (merging branches) or an element from the
        //   `proof` array.
        for (uint256 i = 0; i < totalHashes; i++) {
            bytes32 a = leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++];
            bytes32 b = proofFlags[i]
                ? (leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++])
                : proof[proofPos++];
            hashes[i] = _hashPair(a, b);
        }

        if (totalHashes > 0) {
            if (proofPos != proofLen) {
                revert MerkleProofInvalidMultiproof();
            }
            unchecked {
                return hashes[totalHashes - 1];
            }
        } else if (leavesLen > 0) {
            return leaves[0];
        } else {
            return proof[0];
        }
    }

    /**
     * @dev Calldata version of {processMultiProof}.
     *
     * CAUTION: Not all Merkle trees admit multiproofs. See {processMultiProof} for details.
     */
    function processMultiProofCalldata(
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32[] memory leaves
    ) internal pure returns (bytes32 merkleRoot) {
        // This function rebuilds the root hash by traversing the tree up from the leaves. The root is rebuilt by
        // consuming and producing values on a queue. The queue starts with the `leaves` array, then goes onto the
        // `hashes` array. At the end of the process, the last hash in the `hashes` array should contain the root of
        // the Merkle tree.
        uint256 leavesLen = leaves.length;
        uint256 proofLen = proof.length;
        uint256 totalHashes = proofFlags.length;

        // Check proof validity.
        if (leavesLen + proofLen != totalHashes + 1) {
            revert MerkleProofInvalidMultiproof();
        }

        // The xxxPos values are "pointers" to the next value to consume in each array. All accesses are done using
        // `xxx[xxxPos++]`, which return the current value and increment the pointer, thus mimicking a queue's "pop".
        bytes32[] memory hashes = new bytes32[](totalHashes);
        uint256 leafPos = 0;
        uint256 hashPos = 0;
        uint256 proofPos = 0;
        // At each step, we compute the next hash using two values:
        // - a value from the "main queue". If not all leaves have been consumed, we get the next leaf, otherwise we
        //   get the next hash.
        // - depending on the flag, either another value from the "main queue" (merging branches) or an element from the
        //   `proof` array.
        for (uint256 i = 0; i < totalHashes; i++) {
            bytes32 a = leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++];
            bytes32 b = proofFlags[i]
                ? (leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++])
                : proof[proofPos++];
            hashes[i] = _hashPair(a, b);
        }

        if (totalHashes > 0) {
            if (proofPos != proofLen) {
                revert MerkleProofInvalidMultiproof();
            }
            unchecked {
                return hashes[totalHashes - 1];
            }
        } else if (leavesLen > 0) {
            return leaves[0];
        } else {
            return proof[0];
        }
    }

    /**
     * @dev Sorts the pair (a, b) and hashes the result.
     */
    function _hashPair(bytes32 a, bytes32 b) private pure returns (bytes32) {
        return a < b ? _efficientHash(a, b) : _efficientHash(b, a);
    }

    /**
     * @dev Implementation of keccak256(abi.encode(a, b)) that doesn't allocate or expand memory.
     */
    function _efficientHash(bytes32 a, bytes32 b) private pure returns (bytes32 value) {
        /// @solidity memory-safe-assembly
        assembly {
            mstore(0x00, a)
            mstore(0x20, b)
            value := keccak256(0x00, 0x40)
        }
    }
}

// File: DN404S_nonBlast.sol


pragma solidity ^0.8.25;





contract DN404S is DN404, Ownable {
    error WhitelistMintNotActive();
    error MerkleRootHasNotBeenInitialized();
    error InvalidMerkleProof();
    error CallerIsNotWhitelisted();
    error AddressHasAlreadyClaimed();
    error MaxSupplyExceeded();
    error MerkleRootCannotBeZero();

    error SaleNotActive();
    error MintLimitExceeded();
    error ValueDoesnotMatchMintPrice();

    /// @dev stores metadata URI of the token
    string private revealedURI;

    /// @dev stores hidden metadata URI used before reveal
    string private preRevealedURI;

    /// @dev stores if the nft metadata is revealed or not - initialized to false by default
    bool public revealed;

    /// @dev stores if the public sale is active or not - initialized to false by default
    bool public saleIsActive;

    /// @dev stores if the whitelist mint is active or not - initialized to false by default
    bool public whitelistMintIsActive;

    /// @dev store if buy 1 get 1 offer is active or not - initialized to false by default
    bool public isBuy1Get1OfferActive;

    /// @dev Mapping that tracks whether or not an address has claimed their whitelist mint
    mapping(address => bool) private whitelistClaimed;

    /// @dev Mapping that tracks whether or not an address has claimed their buy 1 get 1 offer
    mapping(address => bool) isBuy1Get1OfferClaimed;

    bytes32 private _merkleRoot_tier1;
    bytes32 private _merkleRoot_tier2;
    bytes32 private _merkleRoot_tier3;

    string public PROVENANCE = "";

    uint256 public mintPrice;
    uint256 public mintLimitPerWallet;
    uint256 public whitelistMintAmount_tier1 = 2;
    uint256 public whitelistMintAmount_tier2 = 5;

    event WhitelistClaimed(address wallet, uint8 tier);
    event TokensMinted(address wallet, uint256 amount);
    event MerkleRootUpdated(bytes32 merkleRoot_, uint8 tier);

    string name_;
    string symbol_;

    constructor(
        string memory _name,
        string memory _symbol,
        uint256 _mintPrice,
        uint256 _mintLimitPerWallet,
        string memory _preRevealedURI,
        string memory _revealedURI,
        bool isRevealed
    )
        Ownable(msg.sender)
    {
        name_ = _name;
        symbol_ = _symbol;
        mintPrice = _mintPrice;
        mintLimitPerWallet = _mintLimitPerWallet;

        if (isRevealed) {
            revealed = true;
            revealedURI = _revealedURI;
        } else {
            preRevealedURI = _preRevealedURI;
        }
    }

    /// @dev Initializes the contract with the initial token supply and the owner of the supply.
    function initialize(address mirror, uint256 _initialTokenSupply) public onlyOwner {
        uint256 initialTokenSupply = _initialTokenSupply *_unit();
        _initializeDN404(initialTokenSupply, msg.sender, mirror);
        (bool success, ) = mirror.call(abi.encodeWithSelector(0x6cef16e6));
        if (!success) {
            revert LinkMirrorContractFailed();
        }
    }

    /// @dev Returns the name of the token.
    function name() public view override(DN404) returns (string memory) {
        return name_;
    }

    /// @dev Returns the symbol of the token.
    function symbol() public view override(DN404) returns (string memory) {
        return symbol_;
    }

    /// function to update the value of mintPrice
    function setMintPrice(uint256 newMintPrice) external onlyOwner {
        mintPrice = newMintPrice;
    }

    /// function to update the value of mintLimitPerWallet
    function setMintLimitPerWallet(uint8 newMintLimitPerWallet) external onlyOwner {
        mintLimitPerWallet = newMintLimitPerWallet;
    }

    /// function to set or update the uri of the metadata (format: ipfs://CID/)
    function setRevealedURI(string memory _revealedURI) public onlyOwner {
        revealedURI = _revealedURI;
    }

    /// function to set or update the pre reveal uri (format: ipfs://CID)
    function setPreRevealedURI(string memory _preRevealedURI) public onlyOwner {
        preRevealedURI = _preRevealedURI;
    }

    // function to set provenance once it's calculated
    function setProvenanceHash(string memory provenanceHash) public onlyOwner {
        PROVENANCE = provenanceHash;
    }

    /// function to reveal the actual metadata once the mint phase is over
    /// call setRevealedURI first
    function reveal() public onlyOwner {
        revealed = true;
    }

    /// pause sale if active, make active if paused
    function flipSaleState() public onlyOwner {
        saleIsActive = !saleIsActive;
    }

    /// pause buy 1 get 1 offer if active, make active if paused
    function flipBuy1Get1Offer() public onlyOwner {
        isBuy1Get1OfferActive = !isBuy1Get1OfferActive;
    }

    // function to mint tokens to specified address by the owner of the contract
    function ownerMint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount*_unit());
        emit TokensMinted(to, amount*_unit());
    }

    /// function to mint an tokens by paying the mint price
    function publicMint(uint256 amount) external payable {
        if (!saleIsActive) {
            revert SaleNotActive();
        }

        if ((balanceOf(msg.sender) + amount*_unit())/_unit() > mintLimitPerWallet) {
            revert MintLimitExceeded();
        }

        if (msg.value < mintPrice * amount) {
            revert ValueDoesnotMatchMintPrice();
        }

        if (!isBuy1Get1OfferClaimed[msg.sender] && isBuy1Get1OfferActive) {
            // offer one free nft if minting for first time
            amount += 1;
            isBuy1Get1OfferClaimed[msg.sender] = true;
        }

        _mint(msg.sender, amount*_unit());
        emit TokensMinted(msg.sender, amount*_unit());
    }

    /// function to mint tokens by whitelisted addresses(tier 1)
    function whitelistMint_tier1(bytes32[] calldata merkleProof_) external {
        if (!whitelistMintIsActive) {
            revert WhitelistMintNotActive();
        }

        bytes32 merkleRootCache = _merkleRoot_tier1;

        if(merkleRootCache == bytes32(0)) {
            revert MerkleRootHasNotBeenInitialized();
        }

        if (whitelistClaimed[msg.sender]) {
            revert AddressHasAlreadyClaimed();
        }

        bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
        if(!MerkleProof.verify(merkleProof_, merkleRootCache, leaf)) {
            revert InvalidMerkleProof();
        }

        whitelistClaimed[_msgSender()] = true;

        _mint(msg.sender, whitelistMintAmount_tier1*_unit());
        emit WhitelistClaimed(msg.sender, 1);
    }

    /// function to mint tokens by whitelisted addresses(tier 2)
    function whitelistMint_tier2(bytes32[] calldata merkleProof_) external {
        if (!whitelistMintIsActive) {
            revert WhitelistMintNotActive();
        }

        bytes32 merkleRootCache = _merkleRoot_tier2;

        if(merkleRootCache == bytes32(0)) {
            revert MerkleRootHasNotBeenInitialized();
        }

        if (whitelistClaimed[msg.sender]) {
            revert AddressHasAlreadyClaimed();
        }

        bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
        if(!MerkleProof.verify(merkleProof_, merkleRootCache, leaf)) {
            revert InvalidMerkleProof();
        }

        whitelistClaimed[_msgSender()] = true;

        _mint(msg.sender, whitelistMintAmount_tier2*_unit());
        emit WhitelistClaimed(msg.sender, 2);
    }

    /// function to mint tokens by whitelisted addresses(tier 3)
    function whitelistMint_tier3(bytes32[] calldata merkleProof_, uint256 amount_) external {
        if (!whitelistMintIsActive) {
            revert WhitelistMintNotActive();
        }

        if ((balanceOf(msg.sender) + amount_*_unit())/_unit() > mintLimitPerWallet) {
            revert MintLimitExceeded();
        }

        bytes32 merkleRootCache = _merkleRoot_tier3;

        if(merkleRootCache == bytes32(0)) {
            revert MerkleRootHasNotBeenInitialized();
        }

        if (whitelistClaimed[msg.sender]) {
            revert AddressHasAlreadyClaimed();
        }

        bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
        if(!MerkleProof.verify(merkleProof_, merkleRootCache, leaf)) {
            revert InvalidMerkleProof();
        }

        whitelistClaimed[_msgSender()] = true;

        _mint(msg.sender, amount_*_unit());
        emit WhitelistClaimed(msg.sender, 3);
    }
    
    /// function to update merkle root(tier 1)
    function setMerkleRoot_tier1(bytes32 merkleRoot_) external onlyOwner {
        if(merkleRoot_ == bytes32(0)) {
            revert MerkleRootCannotBeZero();
        }

        _merkleRoot_tier1 = merkleRoot_;

        emit MerkleRootUpdated(merkleRoot_, 1);
    }

    /// function to update merkle root(tier 2)
    function setMerkleRoot_tier2(bytes32 merkleRoot_) external onlyOwner {
        if(merkleRoot_ == bytes32(0)) {
            revert MerkleRootCannotBeZero();
        }

        _merkleRoot_tier2 = merkleRoot_;

        emit MerkleRootUpdated(merkleRoot_, 2);
    }

    /// function to update merkle root(tier 3)
    function setMerkleRoot_tier3(bytes32 merkleRoot_) external onlyOwner {
        if(merkleRoot_ == bytes32(0)) {
            revert MerkleRootCannotBeZero();
        }

        _merkleRoot_tier3 = merkleRoot_;

        emit MerkleRootUpdated(merkleRoot_, 3);
    }

    /// @notice Returns the merkle root(tier 1)
    function getMerkleRoot_tier1() external view returns (bytes32) {
        return _merkleRoot_tier1;
    }

    /// @notice Returns the merkle root(tier 2)
    function getMerkleRoot_tier2() external view returns (bytes32) {
        return _merkleRoot_tier2;
    }

    /// @notice Returns the merkle root(tier 3)
    function getMerkleRoot_tier3() external view returns (bytes32) {
        return _merkleRoot_tier3;
    }

    /// @notice Returns true if the account already claimed their whitelist mint, false otherwise
    function isWhitelistClaimed(address account) external view returns (bool) {
        return whitelistClaimed[account];
    }

    /// function to pause whitelist mint if active, make active if paused
    function flipWhitelistMintState() public onlyOwner {
        whitelistMintIsActive = !whitelistMintIsActive;
    }

    /// function to set/update the whitelistMintLimit(tier 1)
    function setWhitelistMintAmount_tier1(uint8 _newWhitelistMintAmount)
        public
        onlyOwner
    {
        whitelistMintAmount_tier1 = _newWhitelistMintAmount;
    }

    /// function to set/update the whitelistMintLimit(tier 2)
    function setWhitelistMintAmount_tier2(uint8 _newWhitelistMintAmount)
        public
        onlyOwner
    {
        whitelistMintAmount_tier2 = _newWhitelistMintAmount;
    }

    /// please note that tokenId starts from 0
    function tokenURI(uint256 tokenId) public view returns (string memory) {
        return _tokenURI(tokenId);
    }

    function _tokenURI(uint256 tokenId)
        internal
        view
        override(DN404)
        returns (string memory)
    {
        if (!revealed) {
            return string(abi.encodePacked(preRevealedURI));
        } else {
            return
                string(
                    abi.encodePacked(
                        revealedURI,
                        Strings.toString(tokenId),
                        ".json"
                    )
                );
        }
    }

    /// function to withdraw the balance the contract to owner
    function withdraw() public onlyOwner {
        uint256 balance = address(this).balance;
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Transfer failed.");
    }
}