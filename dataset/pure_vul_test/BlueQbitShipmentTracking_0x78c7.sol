// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BlueQbitShipmentTracking {

    enum Status { Created, PickedUp, InTransit, Delivered, Cancelled }
    Status public shipmentStatus;

    address public shipper;
    address public carrier;

    struct UpdateLog {
        Status status;
        uint256 timestamp;
        address updatedBy;
    }

    UpdateLog[] public updates;

    event StatusUpdated(Status newStatus, uint256 timestamp, address updatedBy);

    modifier onlyAuthorized() {
        require(msg.sender == shipper || msg.sender == carrier, "Non autorizzato");
        _;
    }

    constructor(address _carrier) {
        shipper = msg.sender;
        carrier = _carrier;
        shipmentStatus = Status.Created;
        _logUpdate(Status.Created, msg.sender);
    }

    function updateStatus(Status _newStatus) public onlyAuthorized {
        require(uint(_newStatus) > uint(shipmentStatus), "Non puoi tornare indietro");

        shipmentStatus = _newStatus;
        _logUpdate(_newStatus, msg.sender);
    }

    function _logUpdate(Status _status, address _by) internal {
        updates.push(UpdateLog({
            status: _status,
            timestamp: block.timestamp,
            updatedBy: _by
        }));

        emit StatusUpdated(_status, block.timestamp, _by);
    }

    function getLatestUpdate() public view returns (Status, uint256, address) {
        UpdateLog memory latest = updates[updates.length - 1];
        return (latest.status, latest.timestamp, latest.updatedBy);
    }

    function getUpdateLog(uint index) public view returns (Status, uint256, address) {
        require(index < updates.length, "Indice non valido");
        UpdateLog memory log = updates[index];
        return (log.status, log.timestamp, log.updatedBy);
    }

    function getUpdatesCount() public view returns (uint) {
        return updates.length;
    }
}